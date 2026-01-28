import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


@dataclass
class LLMResponse:
    provider: str
    content: str
    elapsed_s: float


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    name: str

    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful teaching assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        start = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed = time.time() - start
        if response.status_code != 200:
            raise ProviderError(f"OpenAI error: {response.status_code} {response.text}")
        content = response.json()["choices"][0]["message"]["content"]
        return LLMResponse(provider=self.name, content=content, elapsed_s=elapsed)


class GLMProvider(BaseProvider):
    name = "glm"

    def __init__(self, api_key: str, model: str = "glm-4-flash") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful teaching assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        start = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed = time.time() - start
        if response.status_code != 200:
            raise ProviderError(f"GLM error: {response.status_code} {response.text}")
        content = response.json()["choices"][0]["message"]["content"]
        return LLMResponse(provider=self.name, content=content, elapsed_s=elapsed)


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful teaching assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        start = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed = time.time() - start
        if response.status_code != 200:
            raise ProviderError(f"Groq error: {response.status_code} {response.text}")
        content = response.json()["choices"][0]["message"]["content"]
        return LLMResponse(provider=self.name, content=content, elapsed_s=elapsed)


def build_providers() -> Dict[str, BaseProvider]:
    providers: Dict[str, BaseProvider] = {}
    if os.getenv("OPENAI_API_KEY"):
        providers["openai"] = OpenAIProvider(os.environ["OPENAI_API_KEY"])
    if os.getenv("GLM_API_KEY"):
        providers["glm"] = GLMProvider(os.environ["GLM_API_KEY"])
    if os.getenv("GROQ_API_KEY"):
        providers["groq"] = GroqProvider(os.environ["GROQ_API_KEY"])
    return providers


def fanout_generate(prompt: str, providers: Dict[str, BaseProvider], max_workers: int = 3) -> List[LLMResponse]:
    if not providers:
        return []
    responses: List[LLMResponse] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(provider.generate, prompt): provider.name for provider in providers.values()}
        for future in as_completed(futures):
            responses.append(future.result())
    return responses


def pick_provider(preferred: str | None, providers: Dict[str, BaseProvider]) -> BaseProvider:
    if preferred and preferred in providers:
        return providers[preferred]
    return next(iter(providers.values()))


def synthesize(prompt: str, sources: Iterable[LLMResponse], provider: BaseProvider) -> LLMResponse:
    source_block = "\n\n".join(
        [f"### {resp.provider}\n{resp.content.strip()}" for resp in sources]
    )
    synth_prompt = (
        "You are a senior lecturer consolidating multiple drafts.\n\n"
        "Combine the sources into a single, high-quality output. "
        "Resolve conflicts, keep the best explanations, and ensure it is concise.\n\n"
        f"{prompt}\n\nSources:\n{source_block}"
    )
    return provider.generate(synth_prompt)
