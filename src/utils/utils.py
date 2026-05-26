import json
import os
import random
from time import sleep
import yaml
from torch import set_float32_matmul_precision
from tqdm import tqdm
from src.models.pipeline import Pipeline

# ocr
from src.models.ocr.pyMuPDF import ExtractTextByPyMuPDF
from src.models.ocr.Mistral import ExtractTextByMistral
# ir
from src.models.ir.QWenSeventyTwoModel import QWenModel
from src.models.ir.DeepseekR1Model import DeepSeekR1Model
from src.models.ir.ThudmGLM import ThudmGLMModel
from src.models.ir.Hunyuan import HunYuanModel
from src.models.ir.Gpt4oMini import GPT4OMINIModel
from src.models.ir.Gpt5 import GPT5Model
from src.models.ir.QwenCoder import QWenCoderModel

# vlm

from src.models.vlm.GeminiModel import GeminiModel
from src.models.vlm.KimiLatest import KimiLatestModel
from src.models.vlm.DeepSeekVL2 import DeepSeekVLModel
from src.models.vlm.QuenMaxModel import QwenMax
from src.models.vlm.HunYuanVLModel import HunYuanVLModel


# evaluation
from src.evaluation.export_metric_tables import export_metric_tables
from src.evaluation.merge_metric_task_files import merge_metric_task_files
from src.evaluation.cleanup_metrics_and_summaries import delete_unwanted_metric_files, delete_summary_json_files,replace_task_names_in_csv
from src.evaluation.evaluations import evaluate_results_from_file, summarize_results


def append_dict_to_json_file(file_path, new_data, append_mode=True):
    """Append or overwrite data to a JSON file

    Args:
    file_path (str): Path to the JSON file
    new_data (dict or list): Data to append/overwrite
    append_mode (bool): True for append, False for overwrite
    """
    # Make sure the directory exists.
    directory = os.path.dirname(file_path)
    if directory:  # Avoid creating the parent directory of the current directory when file_path is the file name
        os.makedirs(directory, exist_ok=True)

    try:
        if append_mode:
            # Append mode: Read the existing content and add it
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Verify the data type
                if not isinstance(data, list):
                    raise ValueError("JSON content is not a list")
                # Add new data
                if isinstance(new_data, dict):
                    data.append(new_data)
                elif isinstance(new_data, list) and all(isinstance(item, dict) for item in new_data):
                    data.extend(new_data)
                else:
                    raise TypeError("new_data must be a dictionary or list of dictionaries")
            except FileNotFoundError:
                # Create a new list when the file does not exist
                data = [new_data] if isinstance(new_data, dict) else new_data
        else:
            # Overwrite mode: Use new data directly
            data = [new_data] if isinstance(new_data, dict) else new_data

        # Write to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        action = "appended" if append_mode else "overwritten"
        print(f"Successfully {action} data to {file_path}")
        return True

    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file (in append mode)")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


class ManageModel:
    def __init__(self, benchmark_folder="benchmarks"):
        self.benchmark_folder = benchmark_folder
        self._config=self._load_config()

        self.save_path = "eval_sets"
        self.all_case_number = 3000  # total cases
        self.iter_number = 5  # number of cases per round
        self.rounds_number = 5  # cases calculate the score once
        self.error_number = 10  # allow persistent error count

    @staticmethod
    def _load_config():
        # read YAML
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    def batch_summarize_eval_sets(self, eval_sets_dir, save_path, evaluate_results_from_file, summarize_results):
        for model_name in os.listdir(eval_sets_dir):
            model_path = os.path.join(eval_sets_dir, model_name)
            if not os.path.isdir(model_path):
                continue

            for filename in os.listdir(model_path):
                if not filename.endswith(".json"):
                    continue

                result_path = os.path.join(model_path, filename)
                data_name = filename[:-5]  # Remove the.json suffix

                try:
                    results = evaluate_results_from_file(result_path)
                    summary = summarize_results(results)
                except Exception as e:
                    print(f"[❌ ERROR] Failed processing {result_path}: {e}")
                    continue

                # Create the save directory of the model
                output_model_dir = os.path.join(save_path, model_name)
                os.makedirs(output_model_dir, exist_ok=True)

                summary_path = os.path.join(output_model_dir, f"{data_name}_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f_summary:
                    json.dump(summary, f_summary, ensure_ascii=False, indent=2)

                # print(f"[✅ Saved] {summary_path}")


    def _run(self, pipeline, model_name):
        all_files = [f for f in os.listdir(self.benchmark_folder) if f.endswith(".json")]

        for number in range(self.all_case_number//self.iter_number):
            for file_name in all_files:
                data_name = file_name.replace(".json", "")
                file_path = os.path.join(self.benchmark_folder, file_name)

                with open(file_path, "r", encoding="utf-8") as f:
                    test_cases = json.load(f)

                already_path = os.path.join(self.save_path, model_name, f"{data_name}.json")
                already_case_id = set()

                if os.path.exists(already_path):
                    with open(already_path, "r", encoding="utf-8") as f:
                        already_case_id = {case.get("case_id") for case in json.load(f)}

                test_cases = [case for case in test_cases if case.get("case_id") not in already_case_id]
                random.shuffle(test_cases)
                test_cases=test_cases[:self.iter_number]

                if not test_cases:
                    continue

                case_iter = iter(test_cases)
                case = next(case_iter)

                # Basic parameters config
                process_number = 0  # schedule
                current_number_errors = 0  # current error number
                result_path = f"{self.save_path}/{model_name}/{data_name}.json"  # Result file path
                error_case_path = f"{self.save_path}/{model_name}/{data_name}_error.json"
                progress_bar = tqdm(desc=f"📄 Processing {data_name} with model {model_name}", total=len(test_cases))

                while True:
                    # break
                    try:
                        # continuous error
                        if current_number_errors > self.error_number:
                            # append_dict_to_json_file(error_case_path, case)
                            case = next(case_iter, None)
                            if case is None:
                                break
                            sleep(60)
                        # model result
                        pred = pipeline.run(case, r"src\data\raw_pdfs")
                        if "error" in pred.get("predicted_answer"):
                            current_number_errors += 1
                            sleep(20)
                            continue
                        else:  # no error
                            # save the results
                            append_dict_to_json_file(result_path, pred)  # append result

                            process_number += 1
                            progress_bar.update(1)  # progress bar update

                            current_number_errors = 0  # Error number resets to zero
                            case = next(case_iter, None)  # next case

                            if not case:  # case finish
                                break

                    except Exception as case_error:
                        current_number_errors += 1  # error add
                        print(f"[❌ ERROR] Case {process_number} failed: {case_error}")


    def mistral_qwen_api(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = QWenModel(api_key=self._config["aliyun_api_key"])  # Completed
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mistral_qwen_api")

    def mupdf_qwen_api(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = QWenModel(api_key=self._config["aliyun_api_key"])  # Completed
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdf_qwen_api")

    def kimi_latest(self):
        kimi = KimiLatestModel(api_key=self._config["kimi_latest_128k"])
        pipeline = Pipeline(vlm_model=kimi)
        self._run(pipeline, "kimi_latest")

    def deepseek_vlm(self):
        deep = DeepSeekVLModel(api_key=self._config['deepseek_r1_api_key'])
        pipeline = Pipeline(vlm_model=deep)
        self._run(pipeline, "deepseek_vlm")
        
    def hunyuan_vlm(self):
        hunyuan = HHunYuanVLModel(api_key=self._config['hunyuan_api_key'])
        pipeline = Pipeline(vlm_model=hunyuan)
        self._run(pipeline, "hunyuan_vlm")
        
    def mupdf_deepseek_api(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = DeepSeekR1Model(api_key=self._config['deepseek_r1_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdf_deepseek_api")
	
    def mistral_deepseek_api(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = DeepSeekR1Model(api_key=self._config['deepseek_r1_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mistral_deepseek_api")

    def mis_gpt40_mini(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = GPT5Model(api_key=self._config["openai_api_key"])  
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mis_gpt40_mini")

    def mupdf_gpt4o_mini(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = GPT5Model(api_key=self._config["openai_api_key"])  
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdf_gpt4o_mini")
        
        
    def mis_gpt5(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = GPT4OMINIModel(api_key=self._config["openai_api_key"])  
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mis_gpt5")

    def mupdf_gpt5(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = GPT4OMINIModel(api_key=self._config["openai_api_key"])  
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdf_gpt5")

    def gemini(self):
        gemini = GeminiModel(api_key=self._config["gemini_api_key"])
        pipeline = Pipeline(vlm_model=gemini)
        self._run(pipeline, "gemini")

    def qwen_max(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = QwenMax(api_key=self._config["aliyun_api_key"])  
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "qwen_max")

    def mis_glm(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = ThudmGLMModel(api_key=self._config['deepseek_r1_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "misThudmGlm")

    def mupdf_glm(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = ThudmGLMModel(api_key=self._config['deepseek_r1_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdfThudmGlm")

    def mis_hunyuan(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = HunYuanModel(api_key=self._config['hunyuan_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mis_hunyuan")

    def mupdf_hunyuan(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = HunYuanModel(api_key=self._config['hunyuan_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdf_hunyuan")

    def mis_qwen_coder(self):
        ocr_model = ExtractTextByMistral(self._config["mistral_api_key"])
        ir_model = QWenCoderModel(api_key=self._config['aliyun_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mis_qwen_coder")

    def mupdf_qwen_coder(self):
        ocr_model = ExtractTextByPyMuPDF()
        ir_model = QWenCoderModel(api_key=self._config['aliyun_api_key'])
        pipeline = Pipeline(ocr_model, ir_model)
        self._run(pipeline, "mupdf_qwen_coder")

