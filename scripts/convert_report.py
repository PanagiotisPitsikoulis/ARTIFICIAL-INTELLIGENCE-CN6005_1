#!/usr/bin/env python3
import subprocess
from pathlib import Path

script_dir = Path(__file__).parent
docs_dir = script_dir.parent / "docs"

input_file = docs_dir / "report.md"
output_file = docs_dir / "report.docx"

subprocess.run(["pandoc", str(input_file), "-o", str(output_file)], check=True)
print(f"Converted {input_file.name} -> {output_file.name}")
