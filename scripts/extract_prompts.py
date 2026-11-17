"""Extracts the default prompts verbatim from the original C++ `prompts.cpp`.

Usage:  python -I scripts/extract_prompts.py <path/to/prompts.cpp> <dest_dir>

Each `reg(prompts.<field>, "<file_name>", R"( ... )");` block becomes `<dest_dir>/<file_name>.md`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r'reg\(prompts\.(\w+),\s*"([\w]+)",\s*R"\((.*?)\)"\);', re.DOTALL)


def main(src: str, dst: str) -> None:
    text = Path(src).read_text(encoding="utf-8")
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    found = PATTERN.findall(text)
    for field, name, body in found:
        (out / f"{name}.md").write_text(body, encoding="utf-8")
        print(f"{field:28s} -> {name}.md ({len(body)} chars)")
    print(f"extracted {len(found)} prompts")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
