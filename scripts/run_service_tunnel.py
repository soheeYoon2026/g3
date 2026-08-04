"""Run a Cloudflare quick tunnel and publish its private AOX service config."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

import boto3


TUNNEL_URL = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudflared", default="/home/ubuntu/cloudflared")
    parser.add_argument("--target", default="http://127.0.0.1:8003")
    parser.add_argument("--token-file", default="/home/ubuntu/g3/.service-token")
    parser.add_argument("--bucket", default="adro-dev-us-east-1-static")
    parser.add_argument("--key", default="_private/g3/inference-service.json")
    args = parser.parse_args()

    process = subprocess.Popen(
        [args.cloudflared, "tunnel", "--url", args.target, "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    published = False
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = TUNNEL_URL.search(line)
        if not match or published:
            continue
        config = {
            "url": match.group(0),
            "token": open(args.token_file).read().strip(),
        }
        boto3.client("s3", region_name="us-east-1").put_object(
            Bucket=args.bucket,
            Key=args.key,
            Body=json.dumps(config).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        print(f"Published G3 tunnel config to s3://{args.bucket}/{args.key}", flush=True)
        published = True
    raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
