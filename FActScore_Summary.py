import os
import csv
import pandas as pd

#------------------------------load the factscore dir-------------------------
RESULT_DIR = "factscore_result"
OUTPUT_SUMMARY = "factscore_result/factscore_summary.csv"


#---------------------calculate the supported number-----------------------------------
rows = []
for filename in os.listdir(RESULT_DIR):
    if filename.lower().endswith(".csv"):
        path = os.path.join(RESULT_DIR, filename)
        df = pd.read_csv(path)
        total_facts = df['num_facts'].sum()
        total_supported = df['num_supported_facts'].sum()
        fraction = 0.0
        if total_facts > 0:
            fraction = total_supported / total_facts
        print(f"[{filename}] supported {total_supported} / {total_facts} = {fraction:.2%}")
        rows.append([filename, total_facts, total_supported, fraction])
with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    writer.writerow(["filename", "total_facts", "total_supported_facts", "factscore_percent"])
    for row in rows:
        filename, tfacts, tsupport, frac = row
        writer.writerow([filename, tfacts, tsupport, f"{frac:.2%}"])
print(f"summary written to {OUTPUT_SUMMARY}.")

