
import os 
import json
import csv
import openpyxl
import re

def export_metric_tables(output_dir: str, save_dir: str):
    def normalize_task_name(name):
        return re.sub(r"\s*\(\d+\)$", "", name).strip()

    model_map = {
            "mupdf_qwen_api": "PyMuPDF4LLM + Qwen-72B",
            "mistral_qwen_api": "MistralOCR + Qwen-72B",
            "deepseek_vlm": "DeepSeek-VL2",
            "mupdfThudmGlm": "PyMuPDF4LLM + THUDM/GLM-4-32B-0414",
            "misThudmGlm": "MistralOCR + THUDM/GLM-4-32B-0414",
            "mupdf_qwen_coder": "PyMuPDF4LLM + Qwen-Coder-Plus",
            "mis_qwen_coder": "MistralOCR + Qwen-Coder-Plus",
            "mupdf_deepseek_api": "PyMuPDF4LLM + Deepseek-R1",
            "mistral_deepseek_api": "MistralOCR + Deepseek-R1",
            "hunyuan_vlm": "Hunyuan-Turbos-Version",
            "mupdf_hunyuan": "PyMuPDF4LLM + Hunyuan-Turbos-Latest",
            "mis_hunyuan": "MistralOCR + Hunyuan-Turbos-Latest",
            "mupdf_gpt4o_mini": "PyMuPDF4LLM + GPT-4o-mini",
            "mis_gpt4o_mini": "MistralOCR + GPT-4o-mini",
            "mupdf_gpt5": "PyMuPDF4LLM + GPT-5",
            "mis_gpt5": "MistralOCR + GPT-5",
            "qwen_max": "Qwen-Max",
            "gemini": "Gemini-2.5-pro",
            "kimi_latest": "Kimi-latest-128k"
        }

    metrics = ["accuracy", "recall", "f1_score", "bleu_score"]
    results_by_metric = {metric: {} for metric in metrics}
    model_names = list(model_map.values())

    for model_key, model_name in model_map.items():
        model_folder = os.path.join(output_dir, model_key)
        if not os.path.isdir(model_folder):
            continue
        for filename in os.listdir(model_folder):
            if filename.endswith("_summary.txt") or filename.endswith("_summary.json"):
                task_base = filename.replace("_summary.txt", "").replace("_summary.json", "")
                task_name = normalize_task_name(task_base)
                filepath = os.path.join(model_folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                        if isinstance(summary, list) and len(summary) > 0 and isinstance(summary[0], dict):
                            summary = summary[0]
                        elif not isinstance(summary, dict):
                            print(f"⚠️ Unexpected format in {filepath}")
                            continue
                    for metric in metrics:
                        if task_name not in results_by_metric[metric]:
                            results_by_metric[metric][task_name] = {}
                        results_by_metric[metric][task_name][model_name] = summary.get(metric, None)
                except Exception as e:
                    print(f"❌ Failed to read {filepath}: {e}")

    os.makedirs(save_dir, exist_ok=True)
    for metric, task_data in results_by_metric.items():
        for task_name, model_values in task_data.items():
            for model in model_names:
                if model not in model_values:
                    model_values[model] = None
            ordered_model_values = {model: model_values[model] for model in model_names}
            json_path = os.path.join(save_dir, f"{metric}_{task_name}.json")
            with open(json_path, "w", encoding="utf-8") as f_json:
                json.dump(ordered_model_values, f_json, indent=2, ensure_ascii=False)
            csv_path = os.path.join(save_dir, f"{metric}_{task_name}.csv")
            with open(csv_path, "w", newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["Task"] + model_names)
                writer.writerow([task_name] + list(ordered_model_values.values()))
            xlsx_path = os.path.join(save_dir, f"{metric}_{task_name}.xlsx")
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(["Task"] + model_names)
            ws.append([task_name] + list(ordered_model_values.values()))
            wb.save(xlsx_path)
            # print(f"✅ Saved: {metric}_{task_name} → json, csv, xlsx")

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

    for metric in metrics:
        combined_results = []
        merged_json = {}
        for task in task_order:
            task_file = os.path.join(save_dir, f"{metric}_{task}.json")
            if os.path.exists(task_file):
                with open(task_file, "r", encoding="utf-8") as f:
                    model_values = json.load(f)
            else:
                model_values = {}
            for model in model_names:
                if model not in model_values:
                    model_values[model] = None
            row = [task]
            row_values = []
            for model in model_names:
                val = model_values.get(model, None)
                row.append(val)
                row_values.append(val)
            combined_results.append(row)
            merged_json[task] = dict(zip(model_names, row_values))

        json_path = os.path.join(save_dir, f"{metric}.json")
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(merged_json, f_json, indent=2, ensure_ascii=False)
        csv_path = os.path.join(save_dir, f"{metric}.csv")
        with open(csv_path, "w", newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Task"] + model_names)
            writer.writerows(combined_results)
        xlsx_path = os.path.join(save_dir, f"{metric}.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Task"] + model_names)
        for row in combined_results:
            ws.append(row)
        wb.save(xlsx_path)
        # print(f"📦 Summary saved: {metric}.json / .csv / .xlsx")
