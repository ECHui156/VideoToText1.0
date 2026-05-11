import json
import os
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "60"))


def call_llm_cleanup(
    api_key: str,
    system_prompt: str,
    user_text: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> str:
    if not api_key or not api_key.strip():
        raise ValueError("API key is required.")
    prompt = (system_prompt or "").strip()
    if not prompt:
        raise ValueError("System prompt is required.")
    text = (user_text or "").strip()
    if not text:
        raise ValueError("Input text is empty.")

    base = (api_base or DEFAULT_API_BASE).rstrip("/")
    model_name = model or DEFAULT_MODEL
    timeout = DEFAULT_TIMEOUT_SEC if timeout_sec is None else timeout_sec

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=f"{base}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"LLM request failed (HTTP {exc.code}). {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed. {exc.reason}") from exc

    try:
        payload = json.loads(body)
        content = payload["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"LLM response parse failed. Raw: {body}") from exc

    if not content or not str(content).strip():
        raise RuntimeError("LLM returned empty content.")
    return str(content).strip()
