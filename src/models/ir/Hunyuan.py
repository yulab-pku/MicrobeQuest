import os
from openai import OpenAI
#
import os
from openai import OpenAI
import json
from src.types.BaseModel import BaseModel
from typing import Dict, Any
import time
# https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope?scm=20140722.S_help%40%40%E6%96%87%E6%A1%A3%40%402833609.S_RQW%40ag0%2BBB2%40ag0%2BBB1%40ag0%2Bos0.ID_2833609-RL_qwen%7EDAS%7E72B-LOC_doc%7EUND%7Eab-OR_ser-PAR1_2102029b17467816726476506db507-V_4-P0_4-P1_0&spm=a2c4g.11186623.help-search.i4

class HunYuanModel(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name="hunyuan",
            base_url="https://api.hunyuan.cloud.tencent.com/v1",
            api_key=api_key
        )
        self.use_model_name="hunyuan-turbos-latest"
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
                extra_body={
                    "enable_enhancement": True,
                },
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
