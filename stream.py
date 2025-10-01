#!/usr/bin/env python3
import os
from collections import defaultdict

root = "/home/p/.cache/huggingface/lerobot/pierre-safe-sentinels-inc/yam-second-run/images"

counts = defaultdict(int)

for dirpath, _, filenames in os.walk(root):
    # dirpath is the full path; we want the relative subfolder name
    rel = os.path.relpath(dirpath, root)
    if rel == ".":
        continue  # skip the root itself
    counts[rel] += len(filenames)

# Print results sorted by subfolder
for subfolder in sorted(counts):
    print(f"{subfolder}: {counts[subfolder]} files")

# Also print a grand total
print("\nTOTAL:", sum(counts.values()), "files")
