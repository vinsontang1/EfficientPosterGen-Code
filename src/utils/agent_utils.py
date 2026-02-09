import os
import time
import logging
import base64
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import openai
from google import genai
from google.genai import types

from transformers import AutoTokenizer
import tiktoken


from camel.configs import ChatGPTConfig, QwenConfig, GeminiConfig

# =========================
# logging
# =========================
logger = logging.getLogger(__name__)

# =========================
# OpenAI 参数白名单（保持不变）
# =========================
OPENAI_ALLOWED_KWARGS = {
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
}

# =========================
# Provider Enum
# =========================
class ProviderType(Enum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "compatible"
    GOOGLE = "google"

# =========================
# LLM Config
# =========================
@dataclass
class LLMConfig:
    provider: ProviderType
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    model_config_dict: Dict[str, Any] = field(default_factory=dict)

# =========================
# Model Factory
# =========================
class ModelFactory:
    @staticmethod
    def _get_camel_default_config(model_name: str) -> Dict[str, Any]:
        name = model_name.lower()

        if "qwen" in name:
            return QwenConfig().as_dict()
        elif "gemini" in name:
            return GeminiConfig().as_dict()
        else:
            return ChatGPTConfig().as_dict()

    @staticmethod
    def create_config(provider: str, model_name: str) -> LLMConfig:
        provider_str = provider.lower()
        base_model_config = ModelFactory._get_camel_default_config(model_name)

        if model_name == "gpt-5-2025-08-07":
            base_model_config['temperature'] = 1
        
        api_key = None
        base_url = None

        if provider_str == "openai":
            provider_enum = ProviderType.OPENAI
            api_key = os.getenv("OPENAI_API_KEY")

        elif provider_str == "openkey":
            provider_enum = ProviderType.OPENAI_COMPATIBLE
            api_key = os.getenv("OPENKEY_API_KEY")
            base_url = "https://openkey.cloud/v1"

        elif provider_str == "siliconflow":
            provider_enum = ProviderType.OPENAI_COMPATIBLE
            api_key = os.getenv("SILICONFLOW_API_KEY")
            base_url = "https://api.siliconflow.cn/v1"

        elif provider_str == "google":
            provider_enum = ProviderType.GOOGLE
            api_key = os.getenv("GOOGLE_API_KEY")
        
        elif provider_str in ["dashscope", "aliyun", "bailian"]:
            provider_enum = ProviderType.OPENAI_COMPATIBLE
            api_key = os.getenv("DASHSCOPE_API_KEY")
            # 地域（北京 / 弗吉尼亚 / 新加坡）
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            # base_url = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
            # base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

        else:
            raise ValueError(f"不支持的厂家: {provider}")

        return LLMConfig(
            provider=provider_enum,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            model_config_dict=base_model_config,
        )

class TokenUtils:
    _HF_TOKENIZER_CACHE = {}

    @classmethod
    def get_hf_tokenizer(cls, model_keyword: str):
        MAPPING = {
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "deepseek": "deepseek-ai/DeepSeek-V2.5",
            "llama": "meta-llama/Meta-Llama-3-8B",
        }

        hf_model_id = None
        for key, value in MAPPING.items():
            if key in model_keyword.lower():
                hf_model_id = value
                break

        if not hf_model_id:
            hf_model_id = "Qwen/Qwen2.5-7B-Instruct"

        if hf_model_id not in cls._HF_TOKENIZER_CACHE:
            logger.info(f"Loading tokenizer: {hf_model_id}")
            cls._HF_TOKENIZER_CACHE[hf_model_id] = AutoTokenizer.from_pretrained(
                hf_model_id, trust_remote_code=True
            )

        return cls._HF_TOKENIZER_CACHE[hf_model_id]

    @classmethod
    def count_text_tokens(cls, text: str, provider: str, model_name: str) -> int:
        try:
            if provider in ["openai", "compatible"]:
                encoding_name = (
                    "o200k_base" if "gpt-4o" in model_name else "cl100k_base"
                )
                encoding = tiktoken.get_encoding(encoding_name)
                return len(encoding.encode(text))

            elif provider == "google":
                # Google token 由 SDK 返回，这里保持原逻辑
                return 0

            else:
                tokenizer = cls.get_hf_tokenizer(model_name)
                return len(tokenizer.encode(text))

        except Exception as e:
            logger.warning(f"Token count failed: {e}, fallback.")
            return len(text) // 4

# =========================
# Agent
# =========================
class Agent:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._token_cache = {}
        self._init_client()

    # ---------- client init ----------
    def _init_client(self):
        if self.config.provider in (
            ProviderType.OPENAI,
            ProviderType.OPENAI_COMPATIBLE,
        ):
            if not self.config.api_key:
                raise RuntimeError("API key is not set")

            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

        elif self.config.provider == ProviderType.GOOGLE:
            if not self.config.api_key:
                raise RuntimeError("GOOGLE_API_KEY is not set")

            self.client = genai.Client(api_key=self.config.api_key)
            
    def _get_cached_text_tokens(self, text_prompt: str) -> Optional[int]:
        # A. 查缓存：如果以前算过这句话，直接返回
        if text_prompt in self._token_cache:
            logger.debug("Cache Hit: Using stored token count.")
            return self._token_cache[text_prompt]

        # B. 没算过，调用 API
        try:
            text_only_contents = [
                types.Content(
                    role="user", 
                    parts=[types.Part(text=text_prompt)]
                )
            ]
            res = self.client.models.count_tokens(
                model=self.config.model_name,
                contents=text_only_contents
            )
            count = res.total_tokens
            
            # C. 存入缓存 (为了防止内存无限膨胀，可以加个简单的长度判断)
            if len(self._token_cache) > 1000: 
                self._token_cache.clear() # 简单粗暴的清理，或者只存最新的
            
            self._token_cache[text_prompt] = count
            logger.debug(f"📡 API Call: Counted tokens ({count})")
            return count
            
        except Exception as e:
            logger.error(f"Token count failed: {e}")
            return None

    # ---------- public step ----------
    def step(self, prompt: str, images_base64: Optional[List[str]] = None) -> dict:
        images_base64 = images_base64 or []
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self.config.provider in (
                    ProviderType.OPENAI,
                    ProviderType.OPENAI_COMPATIBLE,
                ):
                    return self._step_openai(prompt, images_base64)

                elif self.config.provider == ProviderType.GOOGLE:
                    return self._step_google(prompt, images_base64)

            except Exception as e:
                logger.warning(
                    f"[{self.config.provider.value}] 调用失败 "
                    f"({attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)

        return {}

    # ---------- OpenAI ----------
    def _step_openai(self, prompt: str, images: List[str]) -> dict:
        model_kwargs = {
            k: v
            for k, v in self.config.model_config_dict.items()
            if k in OPENAI_ALLOWED_KWARGS
        }

        if not images:
            messages = [{"role": "user", "content": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                })
            messages = [{"role": "user", "content": content}]

        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **model_kwargs,
        )

        usage = getattr(resp, "usage", None)

        text_input = TokenUtils.count_text_tokens(
            prompt,
            self.config.provider.value,
            self.config.model_name,
        )

        if usage:
            total_input = getattr(usage, "prompt_tokens", text_input)
            output_tokens = getattr(usage, "completion_tokens", 0)
        else:
            total_input = text_input
            output_tokens = 0

        image_input = max(0, total_input - text_input)

        return {
            "content": resp.choices[0].message.content,
            "usage": {
                "input_text": text_input,
                "input_image": image_input,
                "input_total": total_input,
                "output": output_tokens,
            },
        }

    # ---------- Google Gemini ----------
    def _step_google(self, prompt: str, images: List[str]) -> dict:
        text_input = self._get_cached_text_tokens(prompt)
        parts = [types.Part(text=prompt)]
        
        for img_b64 in images:
            try:
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=base64.b64decode(img_b64),
                        )
                    )
                )
            except Exception as e:
                logger.error(f"Failed to decode image: {e}")
                continue
        
        contents = [
            types.Content(
                role="user",
                parts=parts,
            )
        ]
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.config.model_config_dict.get("temperature", 0.7),
                max_output_tokens=self.config.model_config_dict.get("max_tokens", 4096),
            ),
        )
        
        usage = response.usage_metadata
        
        total_input = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        # 精确的图像 token = 总输入 - 精确的文本 token
        if text_input is not None:
             image_input = max(0, total_input - text_input)
        else:
             # 如果缓存或API失败，回退策略
             image_input = 0 
             text_input = total_input
        
        logger.debug(
            f"Token breakdown - Text: {text_input}, Image: {image_input}, "
            f"Total: {total_input}, Output: {output_tokens}"
        )
        
        return {
            "content": response.text,
            "usage": {
                "input_text": text_input, 
                "input_image": image_input,    
                "input_total": total_input,    
                "output": output_tokens,       
            },
        }
