#!/usr/bin/env python3
"""Generate realistic synthetic example data for Oregon Green 488 and pHrodo Red."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phfit.models import sigmoid

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "example")

np.random.seed(2024)

# === Oregon Green 488 (ascending, pKa=4.7) ===
og_pH = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
og_ymin, og_ymax, og_pKa, og_n = 105.0, 845.0, 4.7, 1.0

rows = []
for pH in og_pH:
    true_val = sigmoid(pH, og_ymin, og_ymax, og_pKa, og_n)
    sd = max(true_val * 0.035, 8.0)
    for _ in range(3):
        val = true_val + np.random.normal(0, sd)
        rows.append({"pH": pH, "value": round(val, 1)})

og_df = pd.DataFrame(rows)
og_df.to_csv(os.path.join(EXAMPLE_DIR, "standard_curve.tsv"), sep="\t", index=False)
print("Oregon Green 488 standard curve:")
print(og_df.groupby("pH")["value"].agg(["mean", "std"]).round(1))
print()

# === pHrodo Red (descending, pKa=6.5) ===
ph_pH = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
ph_ymin, ph_ymax, ph_pKa, ph_n = 180.0, 4800.0, 6.5, -1.0

rows = []
for pH in ph_pH:
    true_val = sigmoid(pH, ph_ymin, ph_ymax, ph_pKa, ph_n)
    sd = max(true_val * 0.045, 30.0)
    for _ in range(3):
        val = true_val + np.random.normal(0, sd)
        rows.append({"pH": pH, "value": round(val, 1)})

ph_df = pd.DataFrame(rows)
ph_df.to_csv(os.path.join(EXAMPLE_DIR, "phrodo_standard_curve.tsv"), sep="\t", index=False)
print("pHrodo Red standard curve:")
print(ph_df.groupby("pH")["value"].agg(["mean", "std"]).round(1))
print()

# === Oregon Green sample data ===
og_samples = [("Control", 4.8), ("DrugA", 3.8), ("DrugB", 6.2), ("DrugC", 4.0)]
rows = []
for name, true_pH in og_samples:
    true_val = sigmoid(true_pH, og_ymin, og_ymax, og_pKa, og_n)
    sd = max(true_val * 0.03, 8.0)
    for _ in range(3):
        val = true_val + np.random.normal(0, sd)
        rows.append({"sample": name, "value": round(val, 1)})
pd.DataFrame(rows).to_csv(os.path.join(EXAMPLE_DIR, "sample.tsv"), sep="\t", index=False)
print("Oregon Green sample data saved.")

# === pHrodo Red sample data ===
ph_samples = [("Control", 6.0), ("DrugA", 4.8), ("DrugB", 7.2), ("DrugC", 5.2)]
rows = []
for name, true_pH in ph_samples:
    true_val = sigmoid(true_pH, ph_ymin, ph_ymax, ph_pKa, ph_n)
    sd = max(true_val * 0.04, 30.0)
    for _ in range(3):
        val = true_val + np.random.normal(0, sd)
        rows.append({"sample": name, "value": round(val, 1)})
pd.DataFrame(rows).to_csv(os.path.join(EXAMPLE_DIR, "phrodo_sample.tsv"), sep="\t", index=False)
print("pHrodo Red sample data saved.")
print("\nAll data files regenerated.")
