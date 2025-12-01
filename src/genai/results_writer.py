import os
import csv
from datetime import datetime
from typing import List, Dict


OUTPUT_ROOT = os.path.join(os.getcwd(), "outputs", "genai_benchmarks")


def save_results_incremental(results: List[Dict], run_name: str, batch_size: int = 20):
    if not results:
        return

    out_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "results.csv")

    header = list(results[0].keys())
    write_header = not os.path.exists(out_path)

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, header, delimiter='|', quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerows(results)
