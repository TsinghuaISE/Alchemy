import os

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from .. import PROJ_ROOT


def _load_env() -> None:
    load_dotenv(PROJ_ROOT.parent / ".env", override=False)
    load_dotenv(override=False)


def _resolve_api_key(explicit_api_key: str | None = None) -> str:
    if explicit_api_key:
        return explicit_api_key

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "未找到 API Key。请在环境变量 OPENAI_API_KEY（或 LLM_API_KEY）中设置，"
        "或在项目根目录 .env 中提供对应字段。"
    )


class LLMClient:

    def __init__(self, model: str, base_url: str, **kwargs) -> None:
        _load_env()

        self.model = model
        self.max_tokens = kwargs.get("max_tokens", 20000)
        self.temperature = kwargs.get("temperature", 0.7)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        api_key = _resolve_api_key(kwargs.get("api_key"))
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=kwargs.get("timeout", 300),
            max_retries=3,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.total_calls += 1
        logger.debug(f"LLM call #{self.total_calls}: "
                     f"system={len(system_prompt)}c, user={len(user_prompt)}c")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            logger.debug(f"tokens: +{response.usage.prompt_tokens}p "
                         f"+{response.usage.completion_tokens}c = {self.total_tokens} total")

        return response.choices[0].message.content or ""

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
