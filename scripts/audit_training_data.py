#!/usr/bin/env python3
import sys
import json
from pathlib import Path

FORBIDDEN_STRINGS = [
    "file://",
    "/home/",
    "/Users/",
    "C:\\",
    "Documents/GitHub",
    "github.com/aws",
    "github.com/alibabacloud",
    "AWS SDK",
    "Alibaba Cloud",
    "Django framework",
    "TensorFlow",
    "SageMaker",
    "handle_text_tool_call",
    "QWENClient",
]

def audit_file(file_path: Path) -> bool:
    print(f"Auditing file: {file_path}")
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        print("Rows checked: 0")
        print("Errors found: 1")
        print("-" * 40)
        return False

    rows_checked = 0
    errors_found = 0
    errors_list = []

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
    except Exception as e:
        print(f"Error: Failed to read file {file_path}: {e}")
        print("Rows checked: 0")
        print("Errors found: 1")
        print("-" * 40)
        return False

    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        
        rows_checked += 1
        
        # 1. Validate JSON
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            errors_found += 1
            errors_list.append(f"Line {line_no}: Invalid JSON: {e}")
            continue

        # 2. Check forbidden strings
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden in line:
                errors_found += 1
                errors_list.append(f"Line {line_no}: Contains forbidden term '{forbidden}'")
                break

    print(f"File name: {file_path.name}")
    print(f"Rows checked: {rows_checked}")
    print(f"Errors found: {errors_found}")
    
    if errors_list:
        print("Error details:")
        for err in errors_list[:20]:
            print(f"  - {err}")
        if len(errors_list) > 20:
            print(f"  - ... and {len(errors_list) - 20} more errors.")
            
    print("-" * 40)
    return errors_found == 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit_training_data.py <jsonl_file1> [jsonl_file2 ...]")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    all_clean = True
    for path in paths:
        if not audit_file(path):
            all_clean = False

    if not all_clean:
        sys.exit(1)
    else:
        print("All files passed validation successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
