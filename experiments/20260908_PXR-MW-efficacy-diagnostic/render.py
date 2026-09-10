"""Render a compact view of the canonical README and bind deliverables."""
from pathlib import Path
import hashlib,json,re
import markdown
E=Path(__file__).resolve().parent
text=(E/'README.md').read_text();chunks=text.split('\n## ')
body=markdown.markdown(chunks[0],extensions=['tables','fenced_code'])
for i,chunk in enumerate(chunks[1:]):
 title,_,rest=chunk.partition('\n'); html=markdown.markdown(rest,extensions=['tables','fenced_code'])
 body+=f'<section><h2>{title}</h2>{html}</section>' if i<2 else f'<details><summary>{title}</summary>{html}</details>'
h=hashlib.sha256(text.encode()).hexdigest()
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta name="readme-sha256" content="{h}"><title>PXR molecular weight, efficacy and curve uncertainty</title><style>body{{font:16px/1.55 system-ui,sans-serif;color:#20303a;max-width:1050px;margin:30px auto;padding:0 20px}}h1{{font-size:28px;line-height:1.2}}h2{{font-size:22px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#f0f5f6}}pre{{overflow:auto;background:#f3f5f6;padding:12px;font-size:12px}}a{{color:#176b83}}details{{border-top:1px solid #ccc;padding:12px 0}}summary{{cursor:pointer;font-weight:600;font-size:20px}}code{{overflow-wrap:anywhere}}</style><main>{body}</main></html>'''
(E/'index.html').write_text(page)
for target in re.findall(r'(?:href|src)="([^"]+)"',page):
 if not target.startswith(('http','#')):assert (E/target).is_file(),target
files=[E/n for n in ['README.md','index.html','protocol.json','run.py','verify.py','render.py','verification.json']]+sorted((E/'artifacts').iterdir())+sorted((E/'documentation').iterdir())
manifest={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(E/'deliverable_hashes.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
print(f'PASS: README-bound HTML, local links, {len(manifest)} deliverable hashes')
