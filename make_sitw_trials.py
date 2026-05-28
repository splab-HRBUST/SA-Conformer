from __future__ import annotations

import argparse
from pathlib import Path


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()


def build_core_core(repo: Path, split: str) -> Path:
    base = repo / "sitw_database.v4" / split
    enroll_list = base / "lists" / "enroll-core.lst"
    key_list = base / "keys" / "core-core.lst"

    if not enroll_list.exists():
        raise FileNotFoundError(str(enroll_list))
    if not key_list.exists():
        raise FileNotFoundError(str(key_list))

    enroll_map: dict[str, str] = {}
    for line in _read_lines(enroll_list):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        model = parts[0]
        rel = parts[1]
        enroll_map[model] = rel

    out_lines: list[str] = []
    missing_models: set[str] = set()

    for line in _read_lines(key_list):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        model, test_rel, tag = parts[0], parts[1], parts[2]
        enroll_rel = enroll_map.get(model)
        if enroll_rel is None:
            missing_models.add(model)
            continue

        label = "1" if tag == "tgt" else "0"
        enroll_path = (base / enroll_rel).resolve()
        test_path = (base / test_rel).resolve()
        out_lines.append(f"{label} {enroll_path} {test_path}")

    if missing_models:
        raise KeyError(f"Missing {len(missing_models)} models in enroll-core.lst, e.g. {sorted(missing_models)[:5]}")

    out_path = repo / "data" / f"sitw_{split}_core-core_trials.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default=".", help="SA-mfa_conformer repository root")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    p1 = build_core_core(repo, "dev")
    p2 = build_core_core(repo, "eval")
    print("written:", p1)
    print("written:", p2)


if __name__ == "__main__":
    main()