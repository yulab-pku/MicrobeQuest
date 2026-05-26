
import os
import csv


def replace_task_names_in_csv(input_csv_path, output_csv_path):
    task_name_map = {
        "Strain Entity Recognition and Normalization": "Strain Entity Recognition and Normalization",
        "Strain Entity Resolution": "Strain Entity Resolution",
        "Strain Taxonomy Extraction": "Strain Taxonomy Extraction",
        "Strain Physiological Characteristic Extraction": "Strain Physiological Characteristic Extraction",
        "Environmental Growth Parameter Extraction": "Environmental Growth Parameter Extraction",
        "Strain Attribute Semantic Categorization": "Strain Attribute Semantic Categorization",
        "Strain Culture Medium and Growth Condition Extraction": "Strain Culture Medium and Growth Condition Extraction",
        "Table-based Strain Attribute Extraction": "Table-based Strain Attribute Extraction",
        "Figure-based Strain Attribute Extraction": "Figure-based Strain Attribute Extraction",
        "Multimodal Strain Attribute Reasoning": "Multimodal Strain Attribute Reasoning",
        "Multi-Entity Attribute Association": "Multi-Entity Attribute Association",
        "Multi-value Priority Resolution": "Multi-value Priority Resolution",
        "Negation and Contrast Relationship Parsing": "Negation and Contrast Relationship Parsing",
        "Logical Condition Reasoning": "Logical Condition Reasoning",
        "Cross-paragraph Entity Tracking": "Cross-paragraph Entity Tracking",
        "Implicit Conclusion Generation": "Implicit Conclusion Generation",
        "Multi-instance Comparative Reasoning": "Multi-instance Comparative Reasoning",
        "Semantic Document Region Extraction": "Semantic Document Region Extraction",
    }

    with open(input_csv_path, 'r', encoding='utf-8') as infile, \
            open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)  # The header line is written directly
            else:
                original_task_name = row[0]
                normalized_name = task_name_map.get(original_task_name, original_task_name)
                row[0] = normalized_name
                writer.writerow(row)

    # print(f"✅ Saved updated CSV to: {output_csv_path}")
# replace_task_names_in_csv("f1_score.csv", "f1_score1.csv")

def delete_summary_json_files(eval_sets_dir="eval_sets"):
    deleted_files = []
    for root, dirs, files in os.walk(eval_sets_dir):
        for file in files:
            if file.endswith("_summary.json"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                deleted_files.append(file_path)
    return deleted_files

def delete_unwanted_metric_files(metric_tables_dir="metric_tables"):
    # keep_files = {"accuracy.csv", "bleu_score.csv", "f1_score.csv"}
    keep_files = {"accuracy_result.csv", "bleu_score_result.csv", "f1_score_result.csv"}
    deleted_files = []
    for file in os.listdir(metric_tables_dir):
        if file not in keep_files:
            file_path = os.path.join(metric_tables_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_files.append(file_path)
    return deleted_files
