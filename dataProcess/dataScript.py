import csv
import json

input_file = "data.csv"
output_file = "data.json"

with open(input_file, mode="r", encoding="utf-8-sig") as f:  # 注意这里
    reader = csv.DictReader(f)
    data = [row for row in reader]

with open(output_file, mode="w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"✅ Conversion complete! JSON saved to {output_file}")
