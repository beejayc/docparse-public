"""Fail if a forbidden distribution is installed (name check; backs up the licence-string gate)."""

import sys
from importlib.metadata import distributions

FORBIDDEN = {
    "pymupdf",
    "fitz",
    "extract-msg",
    "mineru",
    "magic-pdf",
    "pdf2image",
    "marker-pdf",
}


def main() -> int:
    installed = {str(d.metadata["Name"]).lower().replace("_", "-") for d in distributions()}
    bad = sorted(installed & FORBIDDEN)
    if bad:
        print(f"forbidden package installed: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
