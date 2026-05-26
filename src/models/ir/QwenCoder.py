#
import os
from openai import OpenAI
import json
from src.types.BaseModel import BaseModel
from typing import Dict, Any
import time
# https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope?scm=20140722.S_help%40%40%E6%96%87%E6%A1%A3%40%402833609.S_RQW%40ag0%2BBB2%40ag0%2BBB1%40ag0%2Bos0.ID_2833609-RL_qwen%7EDAS%7E72B-LOC_doc%7EUND%7Eab-OR_ser-PAR1_2102029b17467816726476506db507-V_4-P0_4-P1_0&spm=a2c4g.11186623.help-search.i4

class QWenCoderModel(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name="qwen-coder",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key
        )
        self.use_model_name="qwen-coder-plus"
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
            # messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': "Context:\n\n\nInstruction:\nList all explicitly stated components included in the medium formulation,the result is a list of ingredients (e.g. ['NaCl','NaHCO3','KH2PO4']), Return a comma-separated list of gas names.Only provide gas names. Do not provide explanations.\n\nNow, answer the following question:\nWhat are the components of the medium?"}, {'role': 'user', 'content': "692: METHANOSALSUM WeN5 MEDIUM\nNaCl\n15.00\ng\nNaHCO3\n10.00\ng\nKH2PO4\n0.30\ng\nNH4Cl\n0.50\ng\nModified Wolin's mineral solution\n10.00\nml\nNiCl2 x 6 H2O (0.1% w/v)\n2.00\nml\nYeast extract\n0.50\ng\nSodium resazurin (0.1% w/v)\n0.50\nml\nNa2CO3\n4.00\ng\nMethanol (50% v/v)\n10.00\nml\nNa2S x 9 H2O\n1.00\ng\nDistilled water\n1000.00\nml\nDissolve ingredients except carbonate, methanol and sulfide, then sparge medium with\n100% N2 gas for 30 - 45 min to make it anoxic. Add and dissolve carbonate and adjust pH\nto 9.0 - 9.2, then dispense medium under 100% N2 gas atmosphere into anoxic\nHungate-type tubes or serum vials and autoclave. Prior to inoculation add methanol (50%\nv/v solution) and sulfide from anoxic stock solutions autoclaved under 100% N2 gas. Adjust\npH of complete medium to 9.2 - 9.4.\nModified Wolin's mineral solution (from medium 141)\nNitrilotriacetic acid\n1.50\ng\nMgSO4 x 7 H2O\n3.00\ng\nMnSO4 x H2O\n0.50\ng\nNaCl\n1.00\ng\nFeSO4 x 7 H2O\n0.10\ng\nCoSO4 x 7 H2O\n0.18\ng\nCaCl2 x 2 H2O\n0.10\ng\nZnSO4 x 7 H2O\n0.18\ng\nCuSO4 x 5 H2O\n0.01\ng\nAlK(SO4)2 x 12 H2O\n0.02\ng\nH3BO3\n0.01\ng\nNa2MoO4 x 2 H2O\n0.01\ng\nNiCl2 x 6 H2O\n0.03\ng\nNa2SeO3 x 5 H2O\n0.30\nmg\nNa2WO4 x 2 H2O\n0.40\nmg\nDistilled water\n1000.00\nml\nFirst dissolve nitrilotriacetic acid and adjust pH to 6.5 with KOH, then add minerals. Adjust\nfinal to pH 7.0 with KOH.\n© 2022 DSMZ - All rights reserved\nPage 1 of 1\n"}]
            # print(f"{type(messages)}, qwen:{messages}")
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