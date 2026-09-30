"""Run the diagnosis service on a local network - no internet connection needed.

The computer that holds the trained model packages serves the web app and the API to phones on
the same network (Wi-Fi router, the computer's own hotspot, or the phone's personal hotspot).
Everything the page needs is served by this computer; nothing is fetched from the internet.

    python lan.py                   # HTTP on port 8000
    python lan.py --https           # HTTPS with a certificate from a local certificate authority
    python lan.py --models-dir D:\\models --port 8080

On start it prints the address(es) to open on the phone, with a QR code for the iPhone camera.

HTTP is enough to take photos and get diagnoses. --https additionally lets the phone install the
page as an app ("Add to Home Screen" with offline start). It creates a local certificate authority
once (lan_certs/ca.crt, valid 10 years) and a server certificate for the current addresses; the
phone must trust the authority once: open https://<address>/lan/ca.crt, install the profile, then
Settings > General > About > Certificate Trust Settings > enable full trust.
"""
import argparse
import datetime
import ipaddress
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def lan_addresses():
    """Private IPv4 addresses of this computer (Wi-Fi / Ethernet / hotspot), best guess first."""
    found = []
    try:  # the interface the OS would route through (works without internet: UDP connect sends nothing)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    result = []
    for ip in found:
        addr = ipaddress.ip_address(ip)
        if (addr.is_private or addr.is_link_local) and not addr.is_loopback and ip not in result:
            result.append(ip)
    return result


def ensure_certificates(cert_dir, ips):
    """Local CA (kept, so the phone trusts it once) + server certificate for the current IPs."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    os.makedirs(cert_dir, exist_ok=True)
    ca_key_path, ca_cert_path = os.path.join(cert_dir, "ca.key"), os.path.join(cert_dir, "ca.crt")
    key_path, cert_path = os.path.join(cert_dir, "server.key"), os.path.join(cert_dir, "server.crt")
    now = datetime.datetime.now(datetime.timezone.utc)

    ca_key = ca_cert = None
    if os.path.exists(ca_key_path) and os.path.exists(ca_cert_path):
        ca_key = serialization.load_pem_private_key(open(ca_key_path, "rb").read(), None)
        ca_cert = x509.load_pem_x509_certificate(open(ca_cert_path, "rb").read())
        try:  # strict TLS clients (iOS, Python 3.13+) need key identifiers; recreate an old CA without them
            ca_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
        except x509.ExtensionNotFound:
            ca_key = ca_cert = None
            for p in (ca_key_path, ca_cert_path, key_path, cert_path):
                if os.path.exists(p):
                    os.remove(p)
    if ca_cert is None:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Cancer Diagnosis Local CA")])
        ca_cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now)
                   .not_valid_after(now + datetime.timedelta(days=3650))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                   .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
                   .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                                content_commitment=False, key_encipherment=False,
                                                data_encipherment=False, key_agreement=False,
                                                encipher_only=False, decipher_only=False), critical=True)
                   .sign(ca_key, hashes.SHA256()))
        open(ca_key_path, "wb").write(ca_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                           serialization.NoEncryption()))
        open(ca_cert_path, "wb").write(ca_cert.public_bytes(serialization.Encoding.PEM))

    wanted = sorted(set(ips) | {"127.0.0.1"})
    if os.path.exists(cert_path):  # reuse the server certificate while the addresses are unchanged
        cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
        have = sorted(str(ip) for ip in cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress))
        if have == wanted and cert.not_valid_after_utc > now + datetime.timedelta(days=7):
            return ca_cert_path, cert_path, key_path

    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname()
    san = [x509.IPAddress(ipaddress.ip_address(ip)) for ip in wanted] + [x509.DNSName("localhost"),
                                                                          x509.DNSName(f"{host}.local")]
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(ca_cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=397))  # Apple's maximum for TLS certificates
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, content_commitment=False,
                                         data_encipherment=False, key_agreement=False, key_cert_sign=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    open(key_path, "wb").write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()))
    open(cert_path, "wb").write(cert.public_bytes(serialization.Encoding.PEM))
    return ca_cert_path, cert_path, key_path


def print_banner(urls, ca_url):
    print("\n" + "=" * 64)
    print(" Cancer image diagnosis - local network mode (no internet needed)")
    print("=" * 64)
    if not urls:
        print(" No local network address found. Connect this computer to the same Wi-Fi as the")
        print(" phone, or turn on a hotspot, then start again.")
        return
    for u in urls:
        print(f"  Open on the phone:  {u}")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(urls[0])
        qr.print_ascii(invert=True)
        print("  Scan the QR code with the iPhone camera.")
    except ImportError:
        pass
    except UnicodeEncodeError:  # console or log file that cannot show block characters: the address above is enough
        print()
    if ca_url:
        print(f"  First time on a phone: install the certificate from {ca_url}")
        print("  then Settings > General > About > Certificate Trust Settings > enable full trust.")
    print("  The phone and this computer must be on the same network. Press Ctrl+C to stop.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true", help="serve HTTPS with a local certificate authority")
    ap.add_argument("--models-dir", default=os.getenv("MODELS_DIR", os.path.join(PROJECT, "models")))
    ap.add_argument("--web-dir", default=os.getenv("WEB_DIR", os.path.join(PROJECT, "web", "dist")))
    args = ap.parse_args()

    os.environ["MODELS_DIR"] = args.models_dir
    os.environ["WEB_DIR"] = args.web_dir
    os.environ.setdefault("MODELS_S3_URI", "")  # offline: never try to reach S3
    if not os.path.isdir(args.web_dir):
        sys.exit(f"web app not found in {args.web_dir} - it is part of the release bundle (web/dist)")

    ips = lan_addresses()
    ssl = {}
    scheme = "http"
    if args.https:
        ca, cert, key = ensure_certificates(os.path.join(PROJECT, "lan_certs"), ips)
        os.environ["LAN_CA_CERT"] = ca
        ssl = {"ssl_certfile": cert, "ssl_keyfile": key}
        scheme = "https"
    port = "" if (scheme, args.port) in (("http", 80), ("https", 443)) else f":{args.port}"
    urls = [f"{scheme}://{ip}{port}" for ip in ips]
    print_banner(urls, f"{urls[0]}/lan/ca.crt" if args.https and urls else "")

    import uvicorn

    sys.path.insert(0, HERE)
    from app.main import app

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
