#
import os
from openai import OpenAI
import json
from src.types.BaseModel import BaseModel
from typing import Dict, Any
import time

class QwenMax(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name="qwen-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key
        )
        self.use_model_name="qwen-max"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_answer(self, test_case: Dict):
        prompt = self.format_prompt(test_case)
        start_time = time.time()
        try:
            messages = [prompt["system"], prompt["user"]]
            messages.append({
                "role": "user",
                "content": str(test_case["test_case"]["input"]["text"]),
            })

            completion = self.client.chat.completions.create(
                model=self.use_model_name,
                messages=messages,
                extra_body={"enable_thinking": False},
            )
            predicted_answer = completion.choices[0].message.content
            usage = completion.usage
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