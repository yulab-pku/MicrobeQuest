from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.utils import *


if __name__ == '__main__':
    model = ManageModel()

    tasks = [
        model.mupdf_gpt5,
        model.mis_gpt5,
        model.hunyuan_vlm,
        model.deepseek_vlm,
        model.gemini,
        model.qwen_max,
        model.mis_glm,
        model.mupdf_glm,
        model.mis_hunyuan,
        model.mupdf_hunyuan,
        model.mis_qwen_coder,
        model.mupdf_qwen_coder,
        model.kimi_latest,
        model.mupdf_deepseek_api,
        model.mistral_deepseek_api,
        model.mis_gpt40_mini,
        model.mupdf_gpt4o_mini,
        model.mupdf_qwen_api,
        model.mistral_qwen_api,

    ]
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjustable max_workers
        futures = {executor.submit(task): task.__name__ for task in tasks}

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                print(f"[✅] {name} completed")
            except Exception as e:
                print(f"[❌ ERROR] {name} failed with error: {e}")

    # Evaluation
    model.batch_summarize_eval_sets(
        eval_sets_dir=model.save_path,
        save_path=model.save_path,  # Save it back to the model folder
        evaluate_results_from_file=evaluate_results_from_file,
        summarize_results=summarize_results
    )

    folder_path = 'Assessment_score_tables'
    export_metric_tables(output_dir=model.save_path, save_dir=folder_path)  # Import as a file
    merge_metric_task_files(folder_path)  # Form the final table
    delete_summary_json_files(eval_sets_dir=model.save_path)  # Delete all files except _summary.json

    # Traverse all the files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            input_path = os.path.join(folder_path, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_result{ext}"
            output_path = os.path.join(folder_path, output_filename)
            replace_task_names_in_csv(input_path, output_path)
    delete_unwanted_metric_files(folder_path)