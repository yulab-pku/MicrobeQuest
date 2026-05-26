import os
import time
import math
import base64
from typing import Dict
from PIL import Image
from openai import OpenAI
from src.types.BaseModel import BaseModel

MAX_TOKENS = 4096  # 模型最大 token
MAX_IMAGE_COUNT = 2  # 最大图片数量
EST_TOKEN_PER_BASE64_KB = 1  # 经验值：1KB base64 ≈ 1333 token
IMAGE_MAX_SIZE = (640,640)  # 可选：图片缩放尺寸，减少 token 消耗


class DeepSeekVLModel(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name="deepseek-ai/deepseek-vl2",
            base_url="https://api.siliconflow.cn/v1/",
            api_key=api_key
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_image_code(self, image_path):
        """
        压缩图片并转换为 base64
        """
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(IMAGE_MAX_SIZE)  # 压缩尺寸
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_image

    def estimate_image_tokens(self, base64_str):
        kb_size = len(base64_str.encode("utf-8")) / 1024
        return math.ceil(kb_size * EST_TOKEN_PER_BASE64_KB)

    def generate_answer(self, test_case: Dict, image_root_path: str):
        prompt = self.format_prompt(test_case)

        # 获取图片路径
        raw_image_indices = test_case['test_case']['input'].get('image_index', [])
        if isinstance(raw_image_indices, str):
            raw_image_indices = [int(x.strip()) for x in raw_image_indices.split(",") if x.strip()]
        image_paths = [os.path.join(image_root_path, f"{img_id}.png") for img_id in raw_image_indices]

        start_time = time.time()
        try:
            # 初始化 messages
            messages = [prompt["system"]]

            # 用户文本内容
            user_content = prompt["user"]["content"]
            text_token_est = len(user_content) // 4  # 粗略估算

            # 处理图片并估算 token
            image_tokens_list = []
            base64_images = []
            for path in image_paths[:MAX_IMAGE_COUNT]:
                base64_code = self.get_image_code(path)
                base64_images.append(base64_code)
                image_tokens_list.append(self.estimate_image_tokens(base64_code))
            total_image_tokens = sum(image_tokens_list)

            # 截断文本保证总 token 不超过 MAX_TOKENS
            allowed_text_tokens = MAX_TOKENS - total_image_tokens
            if allowed_text_tokens <= 0:
                print("[WARN] Too many images, removing some images...")
                # 按顺序保留图片直到 token 剩余足够
                kept_images = []
                allowed_text_tokens = MAX_TOKENS
                for b64, img_tokens, path in zip(base64_images, image_tokens_list, image_paths[:MAX_IMAGE_COUNT]):
                    if allowed_text_tokens - img_tokens > 0:
                        kept_images.append((b64, path))
                        allowed_text_tokens -= img_tokens
                base64_images = [b64 for b64, _ in kept_images]
                image_paths = [p for _, p in kept_images]

            # 截断文本
            if text_token_est > allowed_text_tokens:
                allowed_chars = allowed_text_tokens * 4
                user_content = user_content[:allowed_chars]

            # 添加文本消息
            messages.append({"role": "user", "content": [{"type": "text", "text": user_content}]})

            # 添加图片消息
            for b64 in base64_images:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                })

            # 调用模型
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
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
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Failed to call model: {e}")
            predicted_answer = "error"
            token_count = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        response_time = time.time() - start_time
        return self.format_result(
            test_case=test_case,
            predicted_answer=predicted_answer,
            prompt=prompt,
            response_time=response_time,
            token_count=token_count
        )