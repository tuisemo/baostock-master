from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Ensure repo root is importable when running from /scripts.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    try:
        import pytest
    except Exception as e:
        print(f"[ERROR] pytest import failed: {e}")
        return 2

    # Forward CLI args to pytest (e.g. -q, -k, etc.)
    return int(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
