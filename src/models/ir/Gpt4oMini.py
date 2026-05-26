import os
import time
import sys
from typing import Dict, Any
import yaml
from openai import OpenAI

from src.types.BaseModel import BaseModel

class GPT4OMINIModel(BaseModel):
    def __init__(self, api_key):
        
        super().__init__(
            model_name="gpt-4o-mini",
            api_key=api_key,
            base_url=''
        )
        self.client = OpenAI(api_key=api_key)


    def generate_answer(self, test_case: Dict) -> Dict[str, Any]:

        prompt = self.format_prompt(test_case)
      
        input_text = test_case['test_case']['input']['text']
        input_prompt = {'role': 'user', 'content': f'TEXT FOR ANALYSIS: {input_text}'}

        start_time = time.time()

        try:
            messages = [prompt["system"], prompt["user"], input_prompt]
            print(messages)


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

