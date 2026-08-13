"""Minimal password-SSH runner for the zzai container (no sshpass/plink on this host).

Usage:
  py -3.13 scripts/zzai.py "<remote shell command>"
  py -3.13 scripts/zzai.py --file local.py /remote/path.py     # upload
  py -3.13 scripts/zzai.py --get /remote/path.json local.json   # download

Credentials come from env so they never sit in the repo:
  ZZAI_HOST, ZZAI_PORT, ZZAI_USER, ZZAI_PASS
Commands are wrapped in `bash -lc` so /etc/profile.d proxy vars are loaded.
"""
from __future__ import annotations

import argparse
import os
import sys

import paramiko


def connect():
    host = os.environ.get("ZZAI_HOST")
    port = int(os.environ.get("ZZAI_PORT", "22"))
    user = os.environ.get("ZZAI_USER", "root")
    pw = os.environ.get("ZZAI_PASS")
    # Host and credentials stay out of the repo: the training box is ephemeral
    # and its address/port change on pod reschedule.
    if not host or not pw:
        raise SystemExit("set ZZAI_HOST / ZZAI_PORT / ZZAI_PASS in the environment first")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=45, banner_timeout=45, auth_timeout=45,
                look_for_keys=False, allow_agent=False)
    return cli


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="")
    ap.add_argument("--file", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    cli = connect()
    try:
        if args.file:
            sftp = cli.open_sftp()
            sftp.put(args.file[0], args.file[1])
            sftp.close()
            print("UPLOADED %s -> %s" % tuple(args.file))
        if args.get:
            sftp = cli.open_sftp()
            sftp.get(args.get[0], args.get[1])
            sftp.close()
            print("DOWNLOADED %s -> %s" % tuple(args.get))
        if args.cmd:
            tr = cli.get_transport()
            ch = tr.open_session(timeout=45)
            ch.settimeout(args.timeout)
            ch.exec_command("bash -lc %s" % _q(args.cmd))
            out = []
            while True:
                if ch.recv_ready():
                    d = ch.recv(65536).decode("utf-8", "replace")
                    out.append(d); sys.stdout.write(d); sys.stdout.flush()
                elif ch.recv_stderr_ready():
                    d = ch.recv_stderr(65536).decode("utf-8", "replace")
                    out.append(d); sys.stderr.write(d); sys.stderr.flush()
                elif ch.exit_status_ready():
                    while ch.recv_ready():
                        d = ch.recv(65536).decode("utf-8", "replace")
                        sys.stdout.write(d)
                    while ch.recv_stderr_ready():
                        d = ch.recv_stderr(65536).decode("utf-8", "replace")
                        sys.stderr.write(d)
                    break
                else:
                    import time
                    time.sleep(0.15)
            rc = ch.recv_exit_status()
            print("\nEXIT=%d" % rc)
            return 0 if rc == 0 else rc
    finally:
        cli.close()
    return 0


def _q(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())