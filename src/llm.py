"""
Единая точка входа к LLM.

Зачем абстракция: у скрипта три возможных бэкенда, и выбор зависит от того,
что доступно на конкретной машине. Вызывающий код не должен об этом знать.

  1. anthropic     — официальный SDK. Нужен ANTHROPIC_API_KEY.
  2. openai_compat — любой OpenAI-совместимый endpoint: OpenRouter, Groq,
                     DeepSeek, GigaChat через прокси, локальная llama.cpp.
                     Нужны LLM_BASE_URL + LLM_API_KEY.
  3. claude_cli    — подпроцесс `claude -p`. Ключ не нужен, работает на
                     подписке Claude. Предварительно один раз:  claude setup-token

Выбор по умолчанию (provider="auto") — первый доступный из списка.

Все ответы кэшируются на диск по sha256 от (модель + system + prompt).
Это не микрооптимизация: прогон базы на 50 компаний — это 50 вызовов,
и при отладке промпта его приходится перезапускать десятки раз. Кэш
делает повторный прогон бесплатным и, что важнее, воспроизводимым.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "llm"


class LLMError(RuntimeError):
    """Бэкенд недоступен или вернул мусор."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    cached: bool


def _find_claude_binary() -> str | None:
    """
    Ищет claude.exe / claude.

    Сначала PATH. Если там нет — десктопное приложение Claude держит
    собственную копию в %APPDATA%\\Claude\\claude-code\\<версия>\\.
    Берём самую свежую версию.
    """
    found = shutil.which("claude")
    if found:
        return found

    candidates: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "Claude" / "claude-code")
    home = Path.home()
    candidates.append(home / ".claude" / "local")

    best: tuple[tuple[int, ...], str] | None = None
    for root in candidates:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            for name in ("claude.exe", "claude"):
                exe = child / name if child.is_dir() else None
                if exe and exe.is_file():
                    # сортируем по версии из имени папки: 2.1.217 > 2.1.99
                    parts = tuple(
                        int(p) for p in re.findall(r"\d+", child.name)
                    ) or (0,)
                    if best is None or parts > best[0]:
                        best = (parts, str(exe))
    return best[1] if best else None


class LLM:
    def __init__(
        self,
        provider: str = "auto",
        model: str | None = None,
        cache_dir: Path = CACHE_DIR,
        timeout: int = 180,
    ):
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = self._resolve_provider(provider)
        self.model = model or self._default_model()

    # --- выбор бэкенда -------------------------------------------------

    def _resolve_provider(self, provider: str) -> str:
        if provider != "auto":
            return provider
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
            return "openai_compat"
        if _find_claude_binary():
            return "claude_cli"
        # Кэш ответов лежит в репозитории, поэтому даже на машине без
        # ключей и без интернета скрипт способен воспроизвести результат.
        if any(self.cache_dir.glob("*.json")):
            return "replay"
        raise LLMError(
            "Не найден ни один рабочий бэкенд LLM.\n"
            "Сделайте что-то одно:\n"
            "  1) export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  2) export LLM_BASE_URL=https://openrouter.ai/api/v1 "
            "и LLM_API_KEY=...\n"
            "  3) установите Claude Code и выполните `claude setup-token`\n"
            "Либо запустите скрипт с --no-llm: факты будут извлечены "
            "детерминированно, без модели."
        )

    def _default_model(self) -> str:
        return {
            "anthropic": "claude-sonnet-5",
            "openai_compat": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            "claude_cli": "",  # CLI сам выбирает модель сессии
        }.get(self.provider, "")

    # --- кэш -----------------------------------------------------------

    def _cache_key(self, system: str, prompt: str) -> Path:
        """
        Ключ считается ТОЛЬКО от system+prompt, без провайдера и модели.

        Так сделано намеренно: иначе запись, сделанная бэкендом claude_cli,
        не нашлась бы при воспроизведении в режиме replay, и весь смысл
        закоммиченного кэша пропал бы. Каким бэкендом получен ответ,
        видно внутри самого файла.
        """
        raw = f"{system}|{prompt}".encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(raw).hexdigest()[:32]}.json"

    # --- публичный API -------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> LLMResponse:
        path = self._cache_key(system, prompt)
        if use_cache and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(data["text"], self.provider, cached=True)

        if self.provider == "replay":
            raise LLMError(
                "режим replay: ответа нет в кэше. Этот промпт не прогонялся "
                "через модель. Нужен реальный бэкенд либо --no-llm."
            )

        fn = {
            "anthropic": self._call_anthropic,
            "openai_compat": self._call_openai_compat,
            "claude_cli": self._call_claude_cli,
        }[self.provider]
        text = fn(prompt, system, max_tokens)

        path.write_text(
            json.dumps(
                {"text": text, "prompt": prompt, "provider": self.provider,
                 "model": self.model},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return LLMResponse(text, self.provider, cached=False)

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        retries: int = 2,
    ) -> dict:
        """
        Просит модель вернуть JSON и разбирает его.

        Модели любят обрамлять JSON пояснениями и markdown-заборами,
        поэтому вытаскиваем первый сбалансированный объект, а не доверяем
        json.loads на сыром ответе. При провале — один повтор без кэша
        с более жёсткой инструкцией.
        """
        hard = system + (
            "\n\nОтвечай ТОЛЬКО валидным JSON-объектом. "
            "Без markdown, без пояснений до или после."
        )
        for attempt in range(retries + 1):
            resp = self.complete(
                prompt, hard, max_tokens, use_cache=(attempt == 0)
            )
            obj = _extract_json(resp.text)
            if obj is not None:
                return obj
        raise LLMError(
            f"Модель не вернула валидный JSON за {retries + 1} попыток. "
            f"Последний ответ: {resp.text[:300]!r}"
        )

    # --- реализации бэкендов -------------------------------------------

    def _call_anthropic(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("pip install anthropic") from e

        client = anthropic.Anthropic(timeout=self.timeout)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or None,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def _call_openai_compat(
        self, prompt: str, system: str, max_tokens: int
    ) -> str:
        import requests

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]
            ),
        }
        r = requests.post(
            os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_claude_cli(self, prompt: str, system: str, max_tokens: int) -> str:
        """
        Гоняет `claude -p` подпроцессом.

        --allowed-tools "" запрещает агенту лезть в файлы и сеть: нам нужна
        чистая генерация текста по уже собранному контексту, а не автономная
        работа. Без этого он начнёт сам ходить в интернет и результат
        перестанет быть воспроизводимым.
        """
        exe = _find_claude_binary()
        if not exe:
            raise LLMError("claude не найден")

        full = f"{system}\n\n{prompt}" if system else prompt
        proc = subprocess.run(
            [exe, "-p", full, "--output-format", "json", "--allowed-tools", ""],
            capture_output=True,
            timeout=self.timeout,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0 and not proc.stdout:
            raise LLMError(f"claude вернул {proc.returncode}: {proc.stderr[:300]}")

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude отдал не JSON: {proc.stdout[:300]!r}") from e

        if data.get("is_error"):
            raise LLMError(f"claude: {data.get('result')}")
        return data.get("result", "")


def _extract_json(text: str) -> dict | None:
    """
    Достаёт первый сбалансированный JSON-объект из текста.

    Считаем скобки, игнорируя те, что внутри строк, — регуляркой это
    надёжно не сделать, потому что вложенность произвольная.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


if __name__ == "__main__":
    # Диагностика: какой бэкенд подхватился и отвечает ли он.
    try:
        llm = LLM()
    except LLMError as e:
        print(e)
        sys.exit(1)
    print(f"провайдер: {llm.provider}  модель: {llm.model or '(по умолчанию)'}")
    r = llm.complete("Ответь ровно одним словом: OK", use_cache=False)
    print(f"ответ: {r.text.strip()[:80]!r}")
