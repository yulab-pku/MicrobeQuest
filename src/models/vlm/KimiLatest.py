import os
from pathlib import Path
from openai import OpenAI
import json
from src.types.BaseModel import BaseModel
from typing import Dict, Any
import time
# kimi_latest_128k

class KimiLatestModel(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name = "kimi-latest-128k",
            base_url = "https://api.moonshot.cn/v1",
            api_key = api_key
        )
        self.use_model_name = "moonshot-v1-128k"
        self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

    def get_file_list(self):
        return self.client.files.list().data
    def delete_file(self, file_id):
        self.client.files.delete(file_id=file_id)

    def generate_answer(self, test_case: Dict, image_root_path: str):
        file_list = self.get_file_list()
        if len(file_list) > 10 :
            for file in file_list:
                self.delete_file(file.id)

        prompt = self.format_prompt(test_case)
        start_time = time.time()
        messages = [prompt["system"], prompt["user"]]

        pdf_index = test_case['test_case']['input']['pdf_index'].split(",")

        try:
            for i in pdf_index:
                pdf_path = os.path.join('src/data/raw_pdfs', f"{i}.pdf")
                file_object = self.client.files.create(file=Path(pdf_path), purpose="file-extract")

                file_content = self.client.files.content(file_id=file_object.id).text
            messages.append({
                "role": "system",
                "content": file_content,
            })


            completion = self.client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=messages,
                temperature=0.3,
            )
            predicted_answer=completion.choices[0].message.content

            token_count = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
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