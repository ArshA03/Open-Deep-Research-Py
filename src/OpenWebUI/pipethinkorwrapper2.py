"""
title: DeepSeek R1
author: zgccrui
description: 在OpwenWebUI中显示DeepSeek R1模型的思维链 - 仅支持0.5.6及以上版本
version: 1.2.13
licence: MIT
"""

import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import asyncio


class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="API endpoint URL (e.g., https://openrouter.ai/api/v1, https://api.openai.com/v1)"
        )
        API_KEY: str = Field(
            default="",
            description="API key for OpenAI/OpenRouter"
        )
        API_MODEL: str = Field(
            default="deepseek/deepseek-r1:free",
            description="Model to use for text generation (e.g., openai/gpt-3.5-turbo, anthropic/claude-2)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.emitter = None

    def pipes(self):
        return [
            {
                "id": self.valves.API_MODEL,
                "name": f"Thinker Wrapper ({self.valves.API_MODEL})",
            }
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        thinking_state = {"thinking": -1}
        self.emitter = __event_emitter__
        stored_references = []

        search_providers = 0
        waiting_for_reference = False

        if not self.valves.API_KEY:
            yield json.dumps({"error": "API key not configured"}, ensure_ascii=False)
            return
        
        headers = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json",
        }
        try:

            model_id = self.valves.API_MODEL
            payload = {
                **body,
                "model": model_id,
                "include_reasoning": True,
                "provider": {
                    "order": [
                        'Targon',
                        'Chutes',
                        'Azure'
                    ],
                    "allow_fallbacks": False
                }
            }

            messages = payload["messages"]
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:

                    alternate_role = (
                        "assistant" if messages[i]["role"] == "user" else "user"
                    )
                    messages.insert(
                        i + 1,
                        {"role": alternate_role, "content": "[Unfinished thinking]"},
                    )
                i += 1


            async with httpx.AsyncClient(http2=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.valves.API_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                ) as response:

                    if response.status_code != 200:
                        error = await response.aread()
                        yield self._format_error(response.status_code, error)
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        json_str = line[len(self.data_prefix) :]

                        if json_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            error_detail = f"Failed to parse - Content: {json_str}, Reason: {e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        if search_providers == 0:
                            stored_references = data.get("references", []) + data.get("citations", [])
                            if stored_references:
                                ref_count = len(stored_references)
                                yield '<details type="search">\n'
                                yield f"<summary>已搜索 {ref_count} 个网站</summary>\n"

                            if data.get("references"):
                                for idx, reference in enumerate(stored_references, 1):
                                    yield f'> {idx}. [{reference["title"]}]({reference["url"]})\n'
                                yield "</details>\n"
                                search_providers = 1
                            # If there are citations in data, it indicates the result is from the PPLX engine
                            elif data.get("citations"):
                                for idx, reference in enumerate(stored_references, 1):
                                    yield f'> {idx}. {reference}\n'
                                yield "</details>\n"
                                search_providers = 2

                        choice = data.get("choices", [{}])[0]
                        
                        state_output = await self._update_thinking_state(
                            choice.get("delta", {}), thinking_state
                        )
                        if state_output:
                            yield state_output
                            if state_output == "<think>":
                                yield "\n"

                        content = self._process_content(choice["delta"])
                        if content:
                            if content.startswith("<think>"):
                                content = re.sub(r"^<think>", "", content)
                                yield "<think>"
                                await asyncio.sleep(0.1)
                                yield "\n"
                            elif content.startswith("</think>"):
                                content = re.sub(r"^</think>", "", content)
                                yield "</think>"
                                await asyncio.sleep(0.1)
                                yield "\n"

                            if search_providers == 1:
                                if "摘要" in content:
                                    waiting_for_reference = True
                                    yield content
                                    continue

                                if waiting_for_reference:
                                    if re.match(r"^(\d+|、)$", content.strip()):
                                        numbers = re.findall(r"\d+", content)
                                        if numbers:
                                            num = numbers[0]
                                            ref_index = int(num) - 1
                                            if 0 <= ref_index < len(stored_references):
                                                ref_url = stored_references[ref_index]["url"]
                                            else:
                                                ref_url = ""
                                            content = f"[[{num}]]({ref_url})"

                                    elif not "摘要" in content:
                                        waiting_for_reference = False
                            elif search_providers == 2:
                                def replace_ref(m):
                                    idx = int(m.group(1)) - 1
                                    if 0 <= idx < len(stored_references):
                                        return f'[[{m.group(1)}]]({stored_references[idx]})'
                                    return f'[[{m.group(1)}]]()'
                                content = re.sub(r'\[(\d+)\]', replace_ref, content)

                            yield content
        except Exception as e:
            yield self._format_exception(e)

    async def _update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        state_output = ""
        if thinking_state["thinking"] == -1 and delta.get("reasoning"):
            thinking_state["thinking"] = 0
            state_output = "<think>"
        elif (
            thinking_state["thinking"] == 0
            and not delta.get("reasoning")
            and delta.get("content")
        ):
            thinking_state["thinking"] = 1
            state_output = "\n</think>\n\n"
        return state_output

    def _process_content(self, delta: dict) -> str:
        return delta.get("reasoning", "") or delta.get("content", "")

    def _emit_status(self, description: str, done: bool = False) -> Awaitable[None]:
        if self.emitter:
            return self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                    },
                }
            )
        return None

    def _format_error(self, status_code: int, error: bytes) -> str:
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")
        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)
