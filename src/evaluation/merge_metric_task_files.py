
import os
import csv
import json
import openpyxl
import re

def merge_metric_task_files(save_dir="metric_tables05157"):
    # Indicator list
    metrics = ["accuracy", "recall", "f1_score", "bleu_score"]

    # Task order (for line order)
    task_order = [
        "Strain Entity Recognition and Normalization",
        "Strain Entity Resolution",
        "Strain Taxonomy Extraction",
        "Strain Physiological Characteristic Extraction",
        "Environmental Growth Parameter Extraction",
        "Strain Attribute Semantic Categorization",
        "Strain Culture Medium and Growth Condition Extraction",
        "Table-based Strain Attribute Extraction",
        "Figure-based Strain Attribute Extraction",
        "Multimodal Strain Attribute Reasoning",
        "Multi-Entity Attribute Association",
        "Multi-value Priority Resolution",
        "Negation and Contrast Relationship Parsing",
        "Logical Condition Reasoning",
        "Cross-Paragraph Entity Tracking",
        "Implicit Conclusion Generation",
        "Multi-instance Comparative Reasoning",
        "Semantic Document Region Extraction",
    ]

    def normalize_task_name(name):
        return ''.join(re.findall(r'[a-zA-Z]', name)).lower()

    def find_matching_csv(metric, task):
        target_key = normalize_task_name(task)
        for fname in os.listdir(save_dir):
            if fname.startswith(metric) and fname.endswith(".csv"):
                task_part = fname[len(metric) + 1:-4]
                if normalize_task_name(task_part) == target_key:
                    return os.path.join(save_dir, fname)
        return None

    for metric in metrics:
        combined_rows = []
        header = None

        for task in task_order:
            csv_file = find_matching_csv(metric, task)
            if csv_file and os.path.exists(csv_file):
                with open(csv_file, newline='', encoding='utf-8') as f:
                    reader = list(csv.reader(f))
                    if not header:
                        header = reader[0]  # The first line is the header
                        combined_rows.append(header)
                    if len(reader) > 1:
                        combined_rows.append(reader[1])  # The second row is the data row
                    else:
                        print(f"⚠️ {csv_file} There are no data rows. Skip")
            else:
                print(f"⚠️ File not found: {metric}_{task}.csv")

        # ==== Save as a merged file ====
        csv_path = os.path.join(save_dir, f"{metric}.csv")
        with open(csv_path, "w", newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerows(combined_rows)

        # ==== Synchronize and save as XLSX ====
        xlsx_path = os.path.join(save_dir, f"{metric}.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = metric
        for row in combined_rows:
            ws.append(row)
        wb.save(xlsx_path)

        # ==== Synchronize and save as JSON ====
        json_data = {}
        if header:
            model_names = header[1:]  # Remove the "Task" column
            for row in combined_rows[1:]:
                task_name = row[0]
                values = row[1:]
                json_data[task_name] = dict(zip(model_names, values))

            json_path = os.path.join(save_dir, f"{metric}.json")
            with open(json_path, "w", encoding='utf-8') as f_json:
                json.dump(json_data, f_json, indent=2, ensure_ascii=False)

        # print(f"✅ Merger completed:{metric}.csv / .xlsx / .json")
