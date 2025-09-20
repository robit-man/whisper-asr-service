#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import shutil
import threading
import itertools
import time
import socket
import ssl
import urllib.request
import errno
import signal
import atexit
import argparse
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ─── Constants ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
VENV_FLAG   = "--in-venv"
VENV_DIR    = os.path.join(SCRIPT_DIR, "venv")
HTTPS_PORT  = 443
PORT_TRIES  = 10  # 443..(443+9)
CERT_DIR    = os.path.join(SCRIPT_DIR, "certs")  # where we store non-LE outputs

# Globals for cleanup
_active_httpd = None
_active_port  = None
_child_proc   = None

# ─── Spinner ─────────────────────────────────────────────────────────────────
class Spinner:
    def __init__(self, msg):
        self.msg   = msg
        self.spin  = itertools.cycle("|/-\\")
        self._stop = threading.Event()
        self._thr  = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self.msg} {next(self.spin)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " "*(len(self.msg)+2) + "\r")
        sys.stdout.flush()
    def __enter__(self): self._thr.start()
    def __exit__(self, exc_type, exc, tb):
        self._stop.set(); self._thr.join()

# ─── Venv bootstrap ──────────────────────────────────────────────────────────
def bootstrap_and_run():
    if VENV_FLAG not in sys.argv:
        if not os.path.isdir(VENV_DIR):
            with Spinner("Creating virtualenv…"):
                subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        pip = os.path.join(VENV_DIR, "Scripts" if os.name=="nt" else "bin", "pip")
        with Spinner("Installing dependencies…"):
            subprocess.check_call([pip, "install", "--upgrade", "pip", "cryptography"])
        py = os.path.join(VENV_DIR, "Scripts" if os.name=="nt" else "bin", "python")
        os.execv(py, [py, __file__, VENV_FLAG] + sys.argv[1:])
    else:
        # remove our flag then continue
        idx = sys.argv.index(VENV_FLAG)
        sys.argv.pop(idx)
        main()

# ─── Config I/O ──────────────────────────────────────────────────────────────
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            return json.load(open(CONFIG_PATH))
        except Exception:
            pass
    # default config
    return {
        "serve_path": os.getcwd(),
        "extra_dns_sans": [],
        "cert_mode": "self",    # self | letsencrypt | stepca | gcpca
        "domains": [],          # e.g. ["example.com","www.example.com"]
        "email": "",            # for LE
        "le_staging": False,    # use Let's Encrypt staging
        # Step CA
        "stepca_url": "",
        "stepca_fingerprint": "",
        "stepca_provisioner": "",
        "stepca_token": "",
        # GCP CA
        "gcpca_pool": "",
        "gcpca_location": "",
        "gcpca_cert_id": ""
    }

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

# ─── Args ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HTTPS dev server with flexible certificate sources.")
    p.add_argument("--cert-mode", choices=["self","letsencrypt","stepca","gcpca"], help="Certificate source/mode.")
    p.add_argument("--domains", help="Comma-separated domain list (for LE/Step/GCP).")
    p.add_argument("--email", help="Email for Let's Encrypt.")
    p.add_argument("--agree-tos", action="store_true", help="Agree to Let's Encrypt TOS (required for LE).")
    p.add_argument("--le-staging", action="store_true", help="Use Let's Encrypt staging environment.")
    p.add_argument("--http01-port", type=int, default=80, help="Port for LE standalone HTTP-01.")
    # Step CA options
    p.add_argument("--stepca-url", help="Step CA URL (for bootstrap).")
    p.add_argument("--stepca-fingerprint", help="Step CA root fingerprint (for bootstrap).")
    p.add_argument("--stepca-provisioner", help="Step CA provisioner name.")
    p.add_argument("--stepca-token", help="Step CA one-time token (optional).")
    # GCP CA options
    p.add_argument("--gcpca-pool", help="GCP Private CA pool ID.")
    p.add_argument("--gcpca-location", help="GCP Private CA location (e.g. us-central1).")
    p.add_argument("--gcpca-cert-id", help="Certificate ID to create (optional; defaults to domain+timestamp).")
    # Misc
    p.add_argument("--renew", action="store_true", help="Run provider-specific renew (LE only) then exit.")
    return p.parse_args()

# ─── Net helpers ─────────────────────────────────────────────────────────────
def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))  # TEST-NET-1, no packets sent
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def get_public_ip(timeout=3):
    try:
        return urllib.request.urlopen("https://api.ipify.org", timeout=timeout).read().decode().strip()
    except Exception:
        return None

def wait_for_listen(port, host="127.0.0.1", timeout_s=8.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False

# ─── Banner ──────────────────────────────────────────────────────────────────
def print_banner(port):
    lan     = get_lan_ip()
    public  = get_public_ip()
    lines = [
        f"  Local : https://{lan}:{port}",
        f"  Public: https://{public}:{port}" if public else "  Public: <none>"
    ]
    w = max(len(l) for l in lines) + 4
    print("\n╔" + "═"*w + "╗")
    for l in lines:
        print("║" + l.ljust(w) + "║")
    print("╚" + "═"*w + "╝\n")

# ─── Self-signed generation ──────────────────────────────────────────────────
def generate_self_signed(cert_file, key_file, extra_dns_sans):
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509 import NameOID, SubjectAlternativeName, DNSName, IPAddress
    import cryptography.x509 as x509
    import ipaddress as ipa

    lan_ip    = get_lan_ip()
    public_ip = get_public_ip()

    keyobj = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    san_list = [DNSName("localhost"), IPAddress(ipa.ip_address("127.0.0.1"))]
    for ip in (lan_ip, public_ip):
        if ip:
            try: san_list.append(IPAddress(ipa.ip_address(ip)))
            except ValueError: pass
    for host in (extra_dns_sans or []):
        host = str(host).strip()
        if host: san_list.append(DNSName(host))

    san  = SubjectAlternativeName(san_list)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, lan_ip)])

    not_before = datetime.now(timezone.utc) - timedelta(minutes=5)
    not_after  = not_before + timedelta(days=365)

    with Spinner("Generating self-signed certificate…"):
        cert = (x509.CertificateBuilder()
                .subject_name(name).issuer_name(name)
                .public_key(keyobj.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(not_before).not_valid_after(not_after)
                .add_extension(san, critical=False)
                .sign(keyobj, hashes.SHA256()))

    with open(key_file, "wb") as f:
        f.write(keyobj.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()))
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

# ─── Signal/Cleanup ──────────────────────────────────────────────────────────
def _cleanup():
    global _active_httpd, _child_proc
    if _active_httpd is not None:
        try: _active_httpd.shutdown()
        except Exception: pass
        try: _active_httpd.server_close()
        except Exception: pass
        _active_httpd = None
    if _child_proc is not None:
        try: _child_proc.terminate()
        except Exception: pass
        try: _child_proc.wait(timeout=3)
        except Exception: pass
        _child_proc = None

atexit.register(_cleanup)

def _signal_handler(signum, frame):
    _cleanup()
    if signum == getattr(signal, "SIGTSTP", None):
        try:
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTSTP)
        except Exception:
            pass
        return
    sys.exit(0)

def install_signal_handlers():
    signal.signal(signal.SIGINT,  _signal_handler)
    if hasattr(signal, "SIGTSTP"): signal.signal(signal.SIGTSTP, _signal_handler)
    if hasattr(signal, "SIGTERM"): signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGHUP"):  signal.signal(signal.SIGHUP,  _signal_handler)
    if hasattr(signal, "SIGQUIT"): signal.signal(signal.SIGQUIT, _signal_handler)

# ─── Server with reuse ───────────────────────────────────────────────────────
class ReusableHTTPSServer(HTTPServer):
    allow_reuse_address = True

def bind_https_server(context, start_port=HTTPS_PORT, tries=PORT_TRIES):
    last_err = None
    for p in range(start_port, start_port + tries):
        try:
            httpd = ReusableHTTPSServer(("0.0.0.0", p), SimpleHTTPRequestHandler)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            return httpd, p
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(f"Unable to bind any port in {start_port}..{start_port+tries-1} (last error: {last_err})")

# ─── Provider: Let's Encrypt (Certbot standalone) ────────────────────────────
def ensure_le_cert(domains, email, agree_tos, staging, http01_port):
    if not domains or not email or not agree_tos:
        raise SystemExit("Let's Encrypt requires --domains, --email and --agree-tos.")
    if shutil.which("certbot") is None:
        raise SystemExit("certbot not found in PATH. Install Certbot for your OS.")
    primary = domains[0]
    live_dir = os.path.join("/etc/letsencrypt/live", primary)
    fullchain = os.path.join(live_dir, "fullchain.pem")
    privkey   = os.path.join(live_dir, "privkey.pem")

    if not (os.path.exists(fullchain) and os.path.exists(privkey)):
        # Ensure port 80 is free for standalone
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", http01_port))
            s.close()
        except OSError:
            print(f"⚠ Port {http01_port} appears busy; Certbot standalone may fail.")
        cmd = ["certbot", "certonly", "--standalone",
               "--non-interactive", "--agree-tos", "-m", email,
               "--preferred-challenges", "http",
               "--http-01-port", str(http01_port)]
        if staging: cmd.append("--staging")
        for d in domains: cmd += ["-d", d]
        with Spinner("Requesting Let's Encrypt certificate…"):
            subprocess.check_call(cmd)

    if not (os.path.exists(fullchain) and os.path.exists(privkey)):
        raise SystemExit("Let's Encrypt did not produce expected files in /etc/letsencrypt/live/<domain>.")
    return os.path.abspath(fullchain), os.path.abspath(privkey)

def renew_le_and_exit():
    if shutil.which("certbot") is None:
        raise SystemExit("certbot not found in PATH.")
    with Spinner("Renewing Let's Encrypt certificates…"):
        subprocess.check_call(["certbot", "renew"])
    print("✔ Renew complete.")
    sys.exit(0)

# ─── Provider: Step CA ───────────────────────────────────────────────────────
def ensure_stepca_cert(domains, url, fp, provisioner, token):
    if shutil.which("step") is None:
        raise SystemExit("step (Smallstep CLI) not found in PATH. Install step.")
    if not domains:
        raise SystemExit("Step CA requires --domains.")
    primary = domains[0]
    out_dir = os.path.join(CERT_DIR, "stepca", primary)
    os.makedirs(out_dir, exist_ok=True)
    cert_file = os.path.join(out_dir, "cert.pem")
    key_file  = os.path.join(out_dir, "key.pem")

    # Bootstrap if URL+fingerprint provided
    if url and fp:
        with Spinner("Bootstrapping Step CA…"):
            subprocess.check_call(["step", "ca", "bootstrap", "--ca-url", url, "--fingerprint", fp])

    cmd = ["step", "ca", "certificate", primary, cert_file, key_file, "--force"]
    if provisioner: cmd += ["--provisioner", provisioner]
    if token:       cmd += ["--token", token]
    with Spinner("Requesting certificate from Step CA…"):
        subprocess.check_call(cmd)

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        raise SystemExit("Step CA did not produce expected cert/key files.")
    return os.path.abspath(cert_file), os.path.abspath(key_file)

# ─── Provider: Google Cloud Private CA ───────────────────────────────────────
def ensure_gcpca_cert(domains, pool, location, cert_id):
    if shutil.which("gcloud") is None:
        raise SystemExit("gcloud CLI not found in PATH. Install and run `gcloud init`.")
    if not domains or not pool or not location:
        raise SystemExit("GCP CA requires --domains, --gcpca-pool, and --gcpca-location.")
    primary = domains[0]
    out_dir = os.path.join(CERT_DIR, "gcpca", primary)
    os.makedirs(out_dir, exist_ok=True)
    cert_file = os.path.join(out_dir, "cert.pem")
    key_file  = os.path.join(out_dir, "key.pem")

    if not cert_id:
        cert_id = f"{primary.replace('.','-')}-{int(time.time())}"

    # This command pattern generates a key and a PEM cert (chain) for the domain.
    # Adjust if your project/org flags are required in your environment.
    cmd = [
        "gcloud", "privateca", "certificates", "create", cert_id,
        f"--issuer-pool={pool}",
        f"--location={location}",
        f"--dns-san={primary}",
        "--generate-key",
        f"--key-output-file={key_file}",
        f"--pem-output-file={cert_file}",
    ]
    # extra SANs
    if len(domains) > 1:
        for d in domains[1:]:
            cmd.append(f"--dns-san={d}")

    with Spinner("Requesting certificate from Google Cloud Private CA…"):
        subprocess.check_call(cmd)

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        raise SystemExit("GCP CA did not produce expected cert/key files.")
    return os.path.abspath(cert_file), os.path.abspath(key_file)

# ─── Resolve certificate source ──────────────────────────────────────────────
def resolve_domains(arg_domains, cfg_domains):
    if arg_domains:
        return [d.strip() for d in arg_domains.split(",") if d.strip()]
    return list(cfg_domains or [])

def ensure_certificates(cert_mode, cfg, args):
    """
    Return (cert_file, key_file) based on chosen mode.
    """
    # default outputs in working dir for self-signed
    if cert_mode == "self":
        cert_file = os.path.join(os.getcwd(), "cert.pem")
        key_file  = os.path.join(os.getcwd(), "key.pem")
        generate_self_signed(cert_file, key_file, cfg.get("extra_dns_sans"))
        return cert_file, key_file

    domains = resolve_domains(args.domains, cfg.get("domains", []))

    if cert_mode == "letsencrypt":
        return ensure_le_cert(
            domains=domains,
            email=(args.email or cfg.get("email") or ""),
            agree_tos=bool(args.agree_tos),
            staging=bool(args.le_staging or cfg.get("le_staging")),
            http01_port=int(args.http01_port),
        )

    if cert_mode == "stepca":
        return ensure_stepca_cert(
            domains=domains,
            url=args.stepca_url or cfg.get("stepca_url") or "",
            fp=args.stepca_fingerprint or cfg.get("stepca_fingerprint") or "",
            provisioner=args.stepca_provisioner or cfg.get("stepca_provisioner") or "",
            token=args.stepca_token or cfg.get("stepca_token") or "",
        )

    if cert_mode == "gcpca":
        return ensure_gcpca_cert(
            domains=domains,
            pool=args.gcpca_pool or cfg.get("gcpca_pool") or "",
            location=args.gcpca_location or cfg.get("gcpca_location") or "",
            cert_id=args.gcpca_cert_id or cfg.get("gcpca_cert_id") or "",
        )

    raise SystemExit(f"Unknown cert mode: {cert_mode}")

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    install_signal_handlers()
    args = parse_args()

    # Need root to bind :443 (Linux) and for LE standalone.
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("⚠ Need root to bind port 443; re-running with sudo…")
        os.execvp("sudo", ["sudo", sys.executable] + sys.argv)

    # Load config then overlay CLI
    cfg = load_config()
    updated = False
    if not os.path.exists(CONFIG_PATH):
        default_path = cfg.get("serve_path") or os.getcwd()
        entered = input(f"Serve path [{default_path}]: ").strip() or default_path
        cfg["serve_path"] = entered
        extra = (input("Extra DNS SANs (comma-separated, optional): ").strip() or "")
        if extra:
            cfg["extra_dns_sans"] = [h.strip() for h in extra.split(",") if h.strip()]
        updated = True

    # Apply CLI overrides into cfg for persistence convenience
    if args.cert_mode:            cfg["cert_mode"] = args.cert_mode; updated = True
    if args.domains:              cfg["domains"]   = resolve_domains(args.domains, cfg.get("domains")); updated = True
    if args.email:                cfg["email"]     = args.email; updated = True
    if args.le_staging:           cfg["le_staging"]= True; updated = True
    if args.stepca_url:           cfg["stepca_url"]= args.stepca_url; updated = True
    if args.stepca_fingerprint:   cfg["stepca_fingerprint"]= args.stepca_fingerprint; updated = True
    if args.stepca_provisioner:   cfg["stepca_provisioner"]= args.stepca_provisioner; updated = True
    if args.stepca_token:         cfg["stepca_token"]= args.stepca_token; updated = True
    if args.gcpca_pool:           cfg["gcpca_pool"]= args.gcpca_pool; updated = True
    if args.gcpca_location:       cfg["gcpca_location"]= args.gcpca_location; updated = True
    if args.gcpca_cert_id:        cfg["gcpca_cert_id"]= args.gcpca_cert_id; updated = True

    if updated: save_config(cfg)

    # Renew-only path (LE)
    if args.renew:
        renew_le_and_exit()

    # cd into serve directory
    os.chdir(cfg["serve_path"])

    # Get cert & key according to mode (may invoke external CLIs)
    cert_mode = cfg.get("cert_mode", "self")
    cert_file, key_file = ensure_certificates(cert_mode, cfg, args)

    # Build SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    # Node app path (if present)
    node_path = shutil.which("node")
    if os.path.exists("package.json") and os.path.exists("server.js") and node_path:
        patch_path = os.path.join(os.getcwd(), "tls_patch.js")
        CERT_ABS = os.path.abspath(cert_file)
        KEY_ABS  = os.path.abspath(key_file)
        with open(patch_path, "w") as f:
            f.write(f"""\
const fs = require('fs');
const https = require('https');
const http = require('http');
const CERT = {json.dumps(CERT_ABS)};
const KEY  = {json.dumps(KEY_ABS)};

// Force HTTPS when app uses http.createServer(...)
const _create = http.createServer;
http.createServer = function (opts, listener) {{
  if (typeof opts === 'function') listener = opts;
  return https.createServer({{ key: fs.readFileSync(KEY), cert: fs.readFileSync(CERT) }}, listener);
}};
const _Server = http.Server;
http.Server = function (...args) {{
  return https.Server({{ key: fs.readFileSync(KEY), cert: fs.readFileSync(CERT) }}, ...args);
}};
http.Server.prototype = _Server.prototype;
""")

        env = os.environ.copy()
        env["PORT"] = str(HTTPS_PORT)  # many apps respect PORT
        cmd = [node_path, "-r", patch_path, "server.js"]

        global _child_proc
        with Spinner(f"Starting Node.js (TLS; target port {HTTPS_PORT})…"):
            _child_proc = subprocess.Popen(cmd, env=env, cwd=os.getcwd())

        # Only print once the port is actually listening
        if wait_for_listen(HTTPS_PORT, host="127.0.0.1", timeout_s=10.0):
            print_banner(HTTPS_PORT)
        else:
            print("⚠ Node app started, but port 443 not detected listening. Check app logs.")
        try:
            _child_proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            _cleanup()
        return

    # Python HTTPS fallback: bind first, THEN print links
    try:
        httpd, port = bind_https_server(context, start_port=HTTPS_PORT, tries=PORT_TRIES)
    except RuntimeError as e:
        raise SystemExit(str(e))

    global _active_httpd, _active_port
    _active_httpd = httpd
    _active_port  = port

    if port != HTTPS_PORT:
        print(f"⚠ Port {HTTPS_PORT} in use; selected free port {port}")

    print(f"→ Serving HTTPS from {os.getcwd()} on 0.0.0.0:{port}")
    print_banner(port)

    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()

if __name__ == "__main__":
    bootstrap_and_run()
