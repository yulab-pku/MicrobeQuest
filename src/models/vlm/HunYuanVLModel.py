import os
import time
from pprint import pprint
from typing import Dict
from openai import OpenAI
import base64
#
# import torch
# from transformers import AutoModelForCausalLM
# from src.models.vlm.deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
# from src.models.vlm.deepseek_vl.utils.io import load_pil_images
from src.types.BaseModel import BaseModel

class HunYuanVLModel(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name="hunyuan-turbos-vision",
            base_url="https://api.hunyuan.cloud.tencent.com/v1",
            api_key=api_key
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_image_code(self, image_path):
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        return base64_image

    def generate_answer(self, test_case: Dict, image_root_path: str):
        prompt = self.format_prompt(test_case)

        # Get image file paths based on the image indices
        # raw_image_indices = test_case['test_case']['input'].get('image_index', '')
        raw_image_indices = test_case['test_case']['input'].get('image_index', [])

        if isinstance(raw_image_indices, list):
            image_paths = [os.path.join(image_root_path, f'{img_id}.png') for img_id in raw_image_indices]
        else:
            image_paths = [os.path.join(image_root_path, f'{raw_image_indices}.png')]

        start_time = time.time()

        try:
            messages = [prompt["system"]]

            user_content = prompt["user"]['content']
            contents = []
            contents.append(
                {
                    "type": "text",
                    "text": user_content}
            )

            for path in image_paths:
                image_code = self.get_image_code(path)
                contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_code}"
                    }
                })
            messages.append({
                "role": "user",
                "content": contents
            })

            completion = self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )

            content = completion.choices[0].message.content

            usage = completion.usage

            predicted_answer = content
            token_count = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }

        except Exception as e:
            print(f"[ERROR] Failed to call model: {e}")
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
