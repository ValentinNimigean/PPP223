"""CLI validation for canonical PPO/DPO preference data."""

from __future__ import annotations

import argparse

from alignment.feedback_schema import load_preferences


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical preference data for PPO-based RLHF.")
    parser.add_argument("path", help="Preference JSONL path.")
    parser.add_argument("--allow-pending", action="store_true", help="Allow pending rows marked in metadata.")
    args = parser.parse_args()

    rows = load_preferences(args.path, allow_pending=args.allow_pending)
    pending = sum(1 for row in rows if row.get("metadata", {}).get("pending"))
    print(f"Validated: {args.path}")
    print(f"rows_checked: {len(rows)}")
    print(f"pending_rows: {pending}")
    print("Preference dataset looks clean.")


if __name__ == "__main__":
    main()
