"""Fail if a Markdown file links to a file in this repository that does not exist."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

broken = []
checked = 0
for md in sorted(ROOT.rglob("*.md")):
    if ".git" in md.parts:
        continue
    in_code = False
    for n, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for target in LINK.findall(line):
            if SCHEME.match(target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            checked += 1
            if not (md.parent / path).exists():
                broken.append(f"{md.relative_to(ROOT)}:{n}: {target}")

if broken:
    print("Broken relative links:")
    print("\n".join(broken))
    sys.exit(1)
print(f"{checked} relative links checked, all resolve.")
