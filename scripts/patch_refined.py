"""
Post-install patch for ReFiNed V1.

ReFiNed V1 has compatibility issues with Windows / Python 3.12+ / transformers 4.48+:
  1. aws.py - strftime("%s") is Unix-only; on Windows it returns "%s" literally,
     causing all S3 downloads to silently produce 0-byte files.
     Fix: replace with .timestamp().
  2. loaders.py - re.compile(" \\(.*\\)$") uses invalid escape sequences,
     producing SyntaxWarning in Python 3.12+ (will become SyntaxError in 3.14).
     Fix: use a raw string.
  3. general_utils.py - add_special_tokens kwarg passed to
     AutoTokenizer.from_pretrained() conflicts with the tokenizer method of the
     same name in transformers >= 4.x.
     Fix: remove the kwarg (both call sites).

Usage:
    python scripts/patch_refined.py          # auto-detects .venv in project root
    python scripts/patch_refined.py --check  # dry-run, only report status
"""

import argparse
import sys
from pathlib import Path

# -- Patch definitions --------------------------------------------------------
# Each patch: (relative path from site-packages, old string, new string, description)
# Patches are applied with str.replace (count=0 means replace ALL occurrences).
PATCHES = [
    (
        "refined/resource_management/aws.py",
        's3_obj.last_modified.strftime("%s")',
        "s3_obj.last_modified.timestamp()",
        "strftime -> timestamp (cross-platform)",
    ),
    (
        "refined/resource_management/loaders.py",
        'title_brackets_pattern = re.compile(" \\(.*\\)$")',
        'title_brackets_pattern = re.compile(r" \\(.*\\)$")',
        "invalid escape -> raw string",
    ),
    (
        "refined/utilities/general_utils.py",
        "            add_special_tokens=add_special_tokens,\n",
        "",
        "drop add_special_tokens kwarg (transformers >= 4.x conflict)",
    ),
    (
        "refined/resource_management/data_lookups.py",
        "            add_special_tokens=False,\n",
        "",
        "drop add_special_tokens kwarg (transformers >= 4.x conflict)",
    ),
]


def find_site_packages() -> Path:
    """Find site-packages inside .venv relative to this script's project root."""
    project_root = Path(__file__).resolve().parent.parent
    # Windows: .venv/Lib/site-packages  |  Unix: .venv/lib/python3.x/site-packages
    candidates = list(project_root.glob(".venv/**/site-packages"))
    if not candidates:
        print("ERROR: no site-packages found in .venv/", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def apply_patches(site_packages: Path, check_only: bool = False) -> bool:
    """Apply all patches. Returns True if all patches are applied (or were already)."""
    all_ok = True

    for rel_path, old, new, desc in PATCHES:
        filepath = site_packages / rel_path
        if not filepath.exists():
            print(f"  SKIP  {rel_path} - file not found (ReFiNed not installed?)")
            all_ok = False
            continue

        content = filepath.read_text(encoding="utf-8")

        if old not in content:
            # Check if already patched (new string present, or old string gone)
            if new == "" or new in content:
                print(f"  OK    {rel_path} - already patched ({desc})")
            else:
                print(f"  WARN  {rel_path} - expected string not found ({desc})")
                print(f"         Looking for: {old!r}")
                all_ok = False
            continue

        if check_only:
            print(f"  NEED  {rel_path} - patch needed ({desc})")
            all_ok = False
            continue

        # Apply patch (replace all occurrences)
        content = content.replace(old, new)
        filepath.write_text(content, encoding="utf-8")
        print(f"  DONE  {rel_path} - patched ({desc})")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Patch ReFiNed V1 for Windows/Python 3.12+")
    parser.add_argument(
        "--check", action="store_true",
        help="Dry-run: report patch status without modifying files",
    )
    args = parser.parse_args()

    site_packages = find_site_packages()
    print(f"Site-packages: {site_packages}\n")

    ok = apply_patches(site_packages, check_only=args.check)

    if args.check and not ok:
        print("\nRun without --check to apply patches.")
        sys.exit(1)
    elif ok:
        print("\nAll patches applied.")
    else:
        print("\nSome patches could not be applied - check warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()