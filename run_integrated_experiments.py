from __future__ import annotations

import json
from pathlib import Path

from integrated_role_formation import run_all_experiments


def main() -> None:
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    summary = run_all_experiments(results_dir)
    print("Integrated experiments completed.")
    print(f"Results directory: {results_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
