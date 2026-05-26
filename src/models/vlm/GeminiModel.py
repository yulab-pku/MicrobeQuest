import os
import base64
import time
from typing import Dict, Any
import google.generativeai as genai
from src.types.BaseModel import BaseModel

# os.environ['https_proxy'] = "http://127.0.0.1:7897"
# os.environ['http_proxy'] = "http://127.0.0.1:7897"
# os.environ['all_proxy'] = "socks5://127.0.0.1:7898"

class GeminiModel(BaseModel):
    def __init__(self, api_key):
        current_model_name = "models/gemini-2.5-pro-preview-03-25"

        super().__init__(
            model_name=current_model_name,
            base_url="https://generativelanguage.googleapis.com/v1",
            api_key=api_key
        )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=current_model_name)

    def get_image_code(self, image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def generate_answer(self, test_case: Dict, image_root_path: str = "") -> Dict[str, Any]:
        prompt = self.format_prompt(test_case)
        start_time = time.time()

        try:
            contents = []

            if "system" in prompt and prompt["system"]["content"]:
                contents.append({"text": prompt["system"]["content"]})

            if "user" in prompt and prompt["user"]["content"]:
                contents.append({"text": prompt["user"]["content"]})

            raw_image_indices = test_case['test_case']['input'].get('image_index', [])
            if isinstance(raw_image_indices, str):
                raw_image_indices = [raw_image_indices]

            for img_idx in raw_image_indices:
                img_path = os.path.join(image_root_path, f"{img_idx}.png")
                if os.path.exists(img_path):
                    image_code = self.get_image_code(img_path)
                    contents.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_code
                        }
                    })

            question = test_case["test_case"].get("question", "")
            input_text = test_case["test_case"]["input"].get("text", question)
            if input_text:
                contents.append({"text": input_text})

            response = self.model.generate_content({"parts": contents})

            predicted_answer = response.text
            token_count = self._estimate_tokens(contents, predicted_answer)

        except Exception as e:
            print(f"[ERROR] Failed to call Gemini model: {e}")
            predicted_answer = "error"
            token_count = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

        response_time = time.time() - start_time

        return self.format_result(
            test_case=test_case,
            predicted_answer=predicted_answer,
            prompt=prompt,
            response_time=response_time,
            token_count=token_count
        )

    def _estimate_tokens(self, contents: list, answer: str) -> Dict[str, int]:
        input_text = " ".join([c["text"] for c in contents if "text" in c])
        return {
            "input_tokens": len(input_text.split()) + len(contents) * 10,
            "output_tokens": len(answer.split()),
            "total_tokens": len(input_text.split()) + len(answer.split()) + len(contents) * 10
        }