"""Markdown + figures folder -> one self-contained HTML with the images embedded.

The document describes confidential geometry, so it must not pass through any
hosting; a single HTML file with base64 images opens anywhere and prints to PDF
from the browser.
"""

import argparse
import base64
import mimetypes
import re
from pathlib import Path

import markdown

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--md", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

text = args.md.read_text(encoding="utf-8")
base = args.md.parent


def embed(match):
    alt, src = match.group(1), match.group(2)
    path = (base / src).resolve()
    if not path.exists():
        return f'<p style="color:#c00">[missing figure: {src}]</p>'
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:{mime};base64,{data}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"],
                         extension_configs={"toc": {"toc_depth": "2-3"}})
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', embed, body)
body = re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', embed, body)

title = re.search(r"^# (.+)$", text, re.M)
title = title.group(1) if title else args.md.stem
css = """
body{font-family:"Noto Sans CJK KR","Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif;
     color:#1c1e24;max-width:1080px;margin:40px auto;padding:0 28px;line-height:1.6;font-size:15.5px}
h1{font-size:30px;border-bottom:2px solid #ddd;padding-bottom:8px}
h2{font-size:23px;margin-top:44px;border-bottom:1px solid #e4e6ea;padding-bottom:4px}
h3{font-size:18px;margin-top:28px}
table{border-collapse:collapse;margin:14px 0;font-size:14px}
th,td{border:1px solid #d6d9de;padding:5px 10px;text-align:left;vertical-align:top}
th{background:#f3f4f6}
code{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:13.5px}
pre{background:#f3f4f6;padding:12px 14px;overflow-x:auto;font-size:13px}
figure{margin:18px 0}figure img{max-width:100%;border:1px solid #e4e6ea}
figcaption{color:#6a6f78;font-size:13px;margin-top:6px}
blockquote{border-left:4px solid #cfd3da;margin:14px 0;padding:4px 14px;color:#444;background:#fafafa}
.toc{background:#f8f9fb;padding:10px 20px;border:1px solid #e4e6ea;font-size:14px}
"""
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{css}</style></head><body>{body}</body></html>"""
args.out.write_text(html, encoding="utf-8")
print(f"HTML: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
