# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT-based solver agent using MCP.

Discovers tools via MCP list_tools, executes via call_tool, and uses
GPT's native tool calling for decision making. Benchmark-agnostic
design - all task-specific information comes from the green agent.

Usage:
    python -m netheal.aaa.gpt_agent --mcp-url http://localhost:9025/mcp
    python -m netheal.aaa.gpt_agent --mcp-url http://localhost:9025/mcp --task-hint "Users report slow connections"

Environment variables:
    - LLM_PROVIDER: azure | openai | anthropic | bedrock (optional, auto-detected if unset)
    - LLM_MODEL: model or deployment name (fallback for provider-specific model vars)
    - AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION / AZURE_OPENAI_DEPLOYMENT
    - OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_ORG_ID
    - ANTHROPIC_API_KEY / ANTHROPIC_MODEL / ANTHROPIC_API_URL
    - AWS_REGION / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN / BEDROCK_MODEL_ID
    - REASONING_EFFORT: low | medium | high (reasoning models)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
try:
    from openai import AzureOpenAI, OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AzureOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore

try:
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore

LOGGER = logging.getLogger("netheal.gpt_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SYSTEM_PROMPT = """You are an AI agent that completes tasks using available tools.

Instructions:
1. Read the task description and objective provided to you carefully
2. Discover available tools and understand what each does from their descriptions
3. Use tools strategically to gather information and complete the objective
4. Be mindful of any step/budget constraints mentioned in your objective
5. Complete the task before running out of steps

IMPORTANT - Response Format:
Before making ANY tool calls, you MUST explain your reasoning in text:
- What you observe from the current state
- What hypothesis you are testing
- Why you chose this specific tool/action
- What you expect to learn from it

This reasoning helps track your diagnostic process. After explaining your reasoning,
then make the appropriate tool call(s).

If you encounter errors indicating no actions are available, immediately attempt to complete 
your objective with the information you have gathered so far.
"""


def _mcp_tool_to_openai_function(tool: Any) -> Dict[str, Any]:
    """Convert MCP tool definition to OpenAI function format."""
    func_def: Dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or f"Tool: {tool.name}",
    }

    if hasattr(tool, "inputSchema") and tool.inputSchema:
        schema = tool.inputSchema
        if isinstance(schema, dict):
            func_def["parameters"] = schema
        else:
            func_def["parameters"] = {"type": "object", "properties": {}}
    else:
        func_def["parameters"] = {"type": "object", "properties": {}, "required": []}

    return {"type": "function", "function": func_def}


@dataclass
class LLMToolCall:
    """Normalized tool call representation across providers."""
    call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class HttpToolSpec:
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class HttpToolListResult:
    tools: List[HttpToolSpec]


@dataclass
class HttpToolContent:
    data: Dict[str, Any]


@dataclass
class HttpToolResult:
    content: List[HttpToolContent]


class HttpToolSession:
    """HTTP fallback session that mirrors the MCP client interface."""

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> HttpToolListResult:
        response = await self._client.get("/tools")
        response.raise_for_status()
        payload = response.json() or {}
        tools: List[HttpToolSpec] = []
        for tool in payload.get("tools", []):
            if not isinstance(tool, dict):
                continue
            tools.append(
                HttpToolSpec(
                    name=str(tool.get("name", "")),
                    description=str(tool.get("description", "")),
                    inputSchema=tool.get("parameters")
                    or {"type": "object", "properties": {}, "required": []},
                )
            )
        return HttpToolListResult(tools=tools)

    async def call_tool(self, name: str, args: Dict[str, Any]) -> HttpToolResult:
        response = await self._client.post(f"/tools/{name}", params=args or {})
        response.raise_for_status()
        data = response.json() or {}
        return HttpToolResult(content=[HttpToolContent(data=data)])


@dataclass
class LLMResponse:
    """Normalized LLM response representation across providers."""
    model: str
    content: Optional[str]
    tool_calls: List[LLMToolCall]
    finish_reason: Optional[str]
    usage: Dict[str, Any]


class GPTAgent:
    """GPT-based solver using MCP and a configurable LLM provider."""

    def __init__(
        self,
        mcp_url: str,
        max_turns: int = 25,
        verbose: bool = False,
        on_event: Optional[callable] = None,
        reasoning_effort: str = "medium",  # low, medium, high - for reasoning models like GPT-5, o3
    ) -> None:
        self.mcp_url = mcp_url.rstrip("/")
        if not self.mcp_url.endswith("/mcp"):
            self.mcp_url = self.mcp_url.rstrip("/") + "/mcp"

        self.max_turns = max_turns
        self.verbose = verbose
        self.on_event = on_event  # Callback for real-time event streaming
        self.reasoning_effort = reasoning_effort  # Controls reasoning depth for reasoning models
        self.enable_http_fallback = os.environ.get("MCP_HTTP_FALLBACK", "1").lower() not in {
            "0",
            "false",
            "no",
        }

        if os.environ.get("MCP_DEBUG"):
            logging.getLogger("mcp").setLevel(logging.DEBUG)
            logging.getLogger("mcp.client").setLevel(logging.DEBUG)
            logging.getLogger("mcp.client.streamable_http").setLevel(logging.DEBUG)
            logging.getLogger("httpx").setLevel(logging.INFO)

        env_path = Path(__file__).parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            LOGGER.info("Loaded environment from %s", env_path)
        else:
            load_dotenv()

        self.provider = self._resolve_provider()
        self.client, self.model = self._init_client()
        self.max_output_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.mcp_tools: Dict[str, Any] = {}
        self.turn_count = 0
        self.task_completed = False

    def _resolve_provider(self) -> str:
        provider = os.environ.get("LLM_PROVIDER")
        if provider:
            provider = provider.strip().lower()
            if provider not in ("azure", "openai", "anthropic", "bedrock"):
                raise RuntimeError(
                    f"Unsupported LLM_PROVIDER '{provider}'. Use azure|openai|anthropic|bedrock."
                )
            return provider

        available = []
        if self._azure_configured():
            available.append("azure")
        if self._openai_configured():
            available.append("openai")
        if self._anthropic_configured():
            available.append("anthropic")
        if self._bedrock_configured():
            available.append("bedrock")

        if not available:
            raise RuntimeError(
                "No LLM provider configured. Set LLM_PROVIDER and the corresponding env vars "
                "(AZURE_OPENAI_*, OPENAI_*, ANTHROPIC_*, or AWS_* for Bedrock)."
            )

        if len(available) == 1:
            return available[0]

        inferred = self._infer_provider_from_model(available)
        if inferred:
            return inferred

        raise RuntimeError(
            "Multiple LLM providers detected. Set LLM_PROVIDER to one of: "
            f"{', '.join(available)}."
        )

    @staticmethod
    def _azure_configured() -> bool:
        return bool(
            os.environ.get("AZURE_OPENAI_ENDPOINT")
            and os.environ.get("AZURE_OPENAI_API_KEY")
            and (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("LLM_MODEL"))
        )

    @staticmethod
    def _openai_configured() -> bool:
        return bool(
            os.environ.get("OPENAI_API_KEY")
            and (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL"))
        )

    @staticmethod
    def _anthropic_configured() -> bool:
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            and (os.environ.get("ANTHROPIC_MODEL") or os.environ.get("LLM_MODEL"))
        )

    @staticmethod
    def _bedrock_configured() -> bool:
        return bool(
            (os.environ.get("BEDROCK_MODEL_ID") or os.environ.get("LLM_MODEL"))
            and (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
        )

    def _infer_provider_from_model(self, available: List[str]) -> Optional[str]:
        model = os.environ.get("LLM_MODEL")
        if not model:
            return None
        lowered = model.lower()
        if "bedrock" in available:
            if lowered.startswith("anthropic.") or lowered.startswith("amazon.") or lowered.startswith("meta."):
                return "bedrock"
        if "anthropic" in available and "claude" in lowered:
            return "anthropic"
        if "openai" in available and "azure" not in available:
            return "openai"
        if "azure" in available and "openai" not in available:
            return "azure"
        return None

    def _init_client(self) -> tuple[Any, str]:
        if self.provider == "azure":
            if AzureOpenAI is None:
                raise RuntimeError("AzureOpenAI SDK not available. Install 'openai>=1.0'.")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("LLM_MODEL")
            if not api_key or not endpoint or not deployment:
                raise RuntimeError("Azure OpenAI environment not configured.")
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            return client, deployment

        if self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not available. Install 'openai>=1.0'.")
            api_key = os.environ.get("OPENAI_API_KEY")
            model = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL")
            base_url = os.environ.get("OPENAI_BASE_URL")
            org_id = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
            if not api_key or not model:
                raise RuntimeError("OpenAI environment not configured.")
            client = OpenAI(api_key=api_key, base_url=base_url, organization=org_id)
            return client, model

        if self.provider == "anthropic":
            if Anthropic is None:
                raise RuntimeError("Anthropic SDK not available. Install 'anthropic>=0.37.0'.")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("LLM_MODEL")
            base_url = os.environ.get("ANTHROPIC_API_URL") or os.environ.get("ANTHROPIC_BASE_URL")
            if not api_key or not model:
                raise RuntimeError("Anthropic environment not configured.")
            client = Anthropic(api_key=api_key, base_url=base_url)
            return client, model

        if self.provider == "bedrock":
            if boto3 is None:
                raise RuntimeError("boto3 not available. Install 'boto3'.")
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            model = os.environ.get("BEDROCK_MODEL_ID") or os.environ.get("LLM_MODEL")
            if not region or not model:
                raise RuntimeError("Bedrock environment not configured.")
            session_token = os.environ.get("AWS_SESSION_TOKEN") or os.environ.get("AWS_SESSION_ID")
            client_kwargs = {"service_name": "bedrock-runtime", "region_name": region}
            access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if access_key and secret_key:
                client_kwargs["aws_access_key_id"] = access_key
                client_kwargs["aws_secret_access_key"] = secret_key
                if session_token:
                    client_kwargs["aws_session_token"] = session_token
            client = boto3.client(**client_kwargs)
            return client, model

        raise RuntimeError(f"Unsupported provider '{self.provider}'.")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the callback if registered."""
        if self.on_event:
            try:
                self.on_event({
                    "type": event_type,
                    "turn": self.turn_count,
                    **data
                })
            except Exception as e:
                LOGGER.warning("Event callback failed: %s", e)

    def _log_mcp_exception(self, exc: Exception) -> None:
        if isinstance(exc, BaseExceptionGroup):
            LOGGER.error(
                "MCP session failed with %d sub-exception(s)", len(exc.exceptions)
            )
            for idx, sub in enumerate(exc.exceptions, start=1):
                LOGGER.error("MCP sub-exception %d: %r", idx, sub)
                LOGGER.error(
                    "MCP sub-exception %d traceback:\n%s",
                    idx,
                    "".join(traceback.format_exception(sub)),
                )
        else:
            LOGGER.exception("MCP session failed")

    async def _load_tools(self, session: Any, source: str) -> bool:
        tools_result = await session.list_tools()
        self.tools = []
        for tool in tools_result.tools:
            self.mcp_tools[tool.name] = tool
            openai_func = _mcp_tool_to_openai_function(tool)
            self.tools.append(openai_func)
            LOGGER.info("  Discovered tool: %s", tool.name)

        if not self.tools:
            LOGGER.error("No tools discovered")
            return False

        LOGGER.info("Discovered %d tools via %s", len(self.tools), source)
        return True

    async def _run_http_fallback(
        self,
        task_hint: Optional[str],
        task_context: Optional[Dict[str, Any]],
        mcp_error: str,
    ) -> Dict[str, Any]:
        base_url = self.mcp_url
        if base_url.endswith("/mcp"):
            base_url = base_url[: -len("/mcp")]
        LOGGER.warning("Falling back to HTTP tools at %s", base_url)

        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            session = HttpToolSession(base_url, client)
            await session.initialize()
            LOGGER.info("HTTP tools session initialized")

            if not await self._load_tools(session, "HTTP"):
                return {"error": "No tools available", "turns": self.turn_count, "mcp_error": mcp_error}

            result = await self._run_loop(session, task_hint, task_context)
            if isinstance(result, dict):
                result.setdefault("mcp_error", mcp_error)
            return result

    async def run(
        self,
        task_hint: Optional[str] = None,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run agent until task completion or max turns.

        Args:
            task_hint: Hint from green agent
            task_context: Additional context from EpisodeStart
        """
        LOGGER.info("Starting GPT agent session")
        LOGGER.info("  MCP URL: %s", self.mcp_url)
        LOGGER.info("  Provider: %s", self.provider)
        LOGGER.info("  Model: %s", self.model)
        LOGGER.info("  Max turns: %d", self.max_turns)

        try:
            async with streamablehttp_client(self.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    LOGGER.info("MCP session initialized")

                    if not await self._load_tools(session, "MCP"):
                        return {"error": "No tools available", "turns": 0}

                    return await self._run_loop(session, task_hint, task_context)

        except Exception as e:
            self._log_mcp_exception(e)
            if not self.enable_http_fallback:
                return {"error": str(e), "turns": self.turn_count}
            return await self._run_http_fallback(task_hint, task_context, str(e))

    async def _run_loop(
        self,
        session: Any,
        task_hint: Optional[str],
        task_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Main tool-calling loop."""
        task_context = task_context or {}
        max_steps = task_context.get("max_steps")
        task_description = task_context.get("task_description")
        objective = task_context.get("objective")

        system_content = SYSTEM_PROMPT
        if task_description or objective or max_steps:
            system_content += "\n\n--- TASK INFORMATION ---\n"
            if task_description:
                system_content += f"\nTASK DESCRIPTION:\n{task_description}\n"
            if objective:
                system_content += f"\nOBJECTIVE:\n{objective}\n"
            if max_steps:
                system_content += f"\nSTEP BUDGET: {max_steps} tool calls maximum\n"

        self.messages = [{"role": "system", "content": system_content}]
        
        # Emit system prompt event
        self._emit_event("system_prompt", {
            "content": system_content,
            "task_description": task_description,
            "objective": objective,
            "max_steps": max_steps,
        })

        user_content = ""
        if task_hint:
            user_content += f"**Hint/Context**: {task_hint}\n\n"

        tool_names = [t["function"]["name"] for t in self.tools]
        user_content += f"**Available Tools**: {', '.join(tool_names)}\n\n"
        user_content += "Begin by exploring the environment and gathering information to complete the task."

        self.messages.append({"role": "user", "content": user_content})
        
        # Emit initial user message event
        self._emit_event("user_message", {
            "content": user_content,
            "hint": task_hint,
            "available_tools": tool_names,
        })

        while self.turn_count < self.max_turns and not self.task_completed:
            self.turn_count += 1
            LOGGER.info("Turn %d/%d", self.turn_count, self.max_turns)
            
            # Emit turn start event
            self._emit_event("turn_start", {"max_turns": self.max_turns})

            response = self._call_llm()
            assistant_message = self._build_assistant_message(response)
            self.messages.append(assistant_message)

            usage_info = response.usage or {}
            
            # Emit LLM response event with content/reasoning
            # Always emit, even if content is empty (model may proceed directly to tool calls)
            llm_content = response.content or ""
            self._emit_event("llm_response", {
                "content": llm_content,
                "has_tool_calls": bool(response.tool_calls),
                "finish_reason": response.finish_reason,
                "model": response.model,
                "usage": usage_info,
                "raw_content_is_none": response.content is None,
                "raw_content_is_empty": response.content == "",
            })

            if response.tool_calls:
                # Emit tool calls event
                tool_call_summaries = [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in response.tool_calls
                ]
                
                self._emit_event("tool_calls", {
                    "tools": tool_call_summaries,
                    "reasoning": llm_content,  # Include reasoning that led to tool calls
                })
                
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments

                    LOGGER.info("  Tool: %s(%s)", tool_name, tool_args)

                    result = await self._execute_tool_mcp(session, tool_name, tool_args)
                    
                    # Emit tool result event
                    self._emit_event("tool_result", {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "result": self._truncate_result(result),
                        "success": "error" not in result,
                    })

                    if self.verbose:
                        LOGGER.info("  Result: %s", str(result)[:500])

                    if self._check_completion(tool_name, result):
                        self.task_completed = True
                        LOGGER.info("Task completed via: %s", tool_name)
                        self._emit_event("task_complete", {
                            "completed_by": tool_name,
                            "total_turns": self.turn_count,
                        })

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    })

                self.messages.extend(tool_results)
            else:
                if response.content:
                    LOGGER.info("  Assistant: %s", response.content[:200])
                    self._emit_event("assistant_message", {
                        "content": response.content,
                    })

                if self._appears_complete(response.content):
                    self.task_completed = True
                    self._emit_event("task_complete", {
                        "completed_by": "natural_completion",
                        "total_turns": self.turn_count,
                    })
                else:
                    self.messages.append({
                        "role": "user",
                        "content": "Please continue using the available tools to complete the task.",
                    })

        return {
            "turns": self.turn_count,
            "task_completed": self.task_completed,
            "max_turns_reached": self.turn_count >= self.max_turns,
        }

    def _build_assistant_message(self, response: LLMResponse) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
        }
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return message

    def _call_llm(self) -> LLMResponse:
        if self.provider in ("azure", "openai"):
            return self._call_openai()
        if self.provider == "anthropic":
            return self._call_anthropic()
        if self.provider == "bedrock":
            return self._call_bedrock()
        raise RuntimeError(f"Unsupported provider '{self.provider}'.")

    def _call_openai(self) -> LLMResponse:
        """Make OpenAI/Azure chat completion call with reasoning support."""
        extra_params = {}
        if self.reasoning_effort:
            extra_params["reasoning_effort"] = self.reasoning_effort
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools if self.tools else None,
            tool_choice="auto" if self.tools else None,
            extra_body=extra_params if extra_params else None,
        )
        
        assistant_message = response.choices[0].message
        tool_calls: List[LLMToolCall] = []
        if assistant_message.tool_calls:
            for tc in assistant_message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    LLMToolCall(call_id=tc.id, name=tc.function.name, arguments=args)
                )

        usage_info: Dict[str, Any] = {}
        usage = response.usage
        if usage:
            usage_info = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            # Reasoning models include reasoning_tokens in completion_tokens_details
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
                if reasoning_tokens:
                    usage_info["reasoning_tokens"] = reasoning_tokens
                    self._emit_event("reasoning_usage", {
                        "reasoning_tokens": reasoning_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "prompt_tokens": usage.prompt_tokens,
                        "reasoning_effort": self.reasoning_effort,
                    })
        
        LOGGER.info("LLM Response - Model: %s", response.model)
        LOGGER.info("LLM Response - Finish reason: %s", response.choices[0].finish_reason)
        LOGGER.info("LLM Response - Content: %s", assistant_message.content)
        LOGGER.info(
            "LLM Response - Tool calls: %s",
            len(assistant_message.tool_calls) if assistant_message.tool_calls else 0,
        )

        return LLMResponse(
            model=response.model,
            content=assistant_message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            usage=usage_info,
        )

    def _call_anthropic(self) -> LLMResponse:
        """Make Anthropic messages call with tool support."""
        system, messages = self._build_anthropic_messages()
        tools = self._openai_tools_to_anthropic() if self.tools else None

        create_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
        }
        if system:
            create_params["system"] = system
        if tools:
            create_params["tools"] = tools

        response = self.client.messages.create(**create_params)

        content_blocks = response.content or []
        text_parts: List[str] = []
        tool_calls: List[LLMToolCall] = []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text_parts.append(getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    LLMToolCall(
                        call_id=getattr(block, "id", None) or (block.get("id") if isinstance(block, dict) else "") or "",
                        name=getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else "") or "",
                        arguments=getattr(block, "input", None) or (block.get("input") if isinstance(block, dict) else {}) or {},
                    )
                )

        usage_info = {}
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        content_text = "".join(text_parts) if text_parts else None
        finish_reason = getattr(response, "stop_reason", None)

        LOGGER.info("LLM Response - Model: %s", response.model)
        LOGGER.info("LLM Response - Finish reason: %s", finish_reason)
        LOGGER.info("LLM Response - Content: %s", content_text)
        LOGGER.info("LLM Response - Tool calls: %s", len(tool_calls))

        return LLMResponse(
            model=response.model,
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage_info,
        )

    def _openai_tools_to_anthropic(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for tool in self.tools:
            function = tool.get("function", {})
            tools.append({
                "name": function.get("name"),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            })
        return tools

    def _call_bedrock(self) -> LLMResponse:
        """Make Bedrock converse call with tool support."""
        system, messages = self._build_bedrock_messages()
        tool_config = None
        if self.tools:
            tool_config = {
                "tools": self._openai_tools_to_bedrock(),
                "toolChoice": {"auto": {}},
            }

        inference_config = {
            "maxTokens": self.max_output_tokens,
            "temperature": float(os.environ.get("LLM_TEMPERATURE", "0")),
        }

        request: Dict[str, Any] = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system:
            request["system"] = [{"text": system}]
        if tool_config:
            request["toolConfig"] = tool_config

        response = self.client.converse(**request)

        output_message = (response.get("output", {}) or {}).get("message", {}) or {}
        content_blocks = output_message.get("content", []) or []

        text_parts: List[str] = []
        tool_calls: List[LLMToolCall] = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block.get("text", ""))
                continue
            if "toolUse" in block:
                tool_use = block.get("toolUse", {}) or {}
                tool_calls.append(
                    LLMToolCall(
                        call_id=tool_use.get("toolUseId", ""),
                        name=tool_use.get("name", ""),
                        arguments=tool_use.get("input", {}) or {},
                    )
                )

        usage_info = {}
        usage = response.get("usage", {}) or {}
        input_tokens = usage.get("inputTokens")
        output_tokens = usage.get("outputTokens")
        if input_tokens is not None and output_tokens is not None:
            usage_info = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        finish_reason = response.get("stopReason")
        content_text = "".join(text_parts) if text_parts else None

        LOGGER.info("LLM Response - Model: %s", self.model)
        LOGGER.info("LLM Response - Finish reason: %s", finish_reason)
        LOGGER.info("LLM Response - Content: %s", content_text)
        LOGGER.info("LLM Response - Tool calls: %s", len(tool_calls))

        return LLMResponse(
            model=self.model,
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage_info,
        )

    def _build_anthropic_messages(self) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for msg in self.messages:
            role = msg.get("role")
            if role == "system":
                system = msg.get("content")
                continue
            if role == "user":
                content = msg.get("content", "")
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": str(content)}],
                })
                continue
            if role == "assistant":
                blocks: List[Dict[str, Any]] = []
                content = msg.get("content")
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for tc in msg.get("tool_calls", []) or []:
                    function = tc.get("function", {})
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": function.get("name", ""),
                        "input": args or {},
                    })
                if blocks:
                    messages.append({"role": "assistant", "content": blocks})
                continue
            if role == "tool":
                tool_id = msg.get("tool_call_id")
                content = msg.get("content", "")
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": str(content),
                        }
                    ],
                })
        return system, messages

    def _openai_tools_to_bedrock(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for tool in self.tools:
            function = tool.get("function", {})
            tools.append({
                "toolSpec": {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "inputSchema": {
                        "json": function.get("parameters", {"type": "object", "properties": {}})
                    },
                }
            })
        return tools

    def _build_bedrock_messages(self) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system: Optional[str] = None
        messages: List[Dict[str, Any]] = []

        for msg in self.messages:
            role = msg.get("role")
            if role == "system":
                system = msg.get("content")
                continue
            if role == "user":
                content = msg.get("content", "")
                messages.append({
                    "role": "user",
                    "content": [{"text": str(content)}],
                })
                continue
            if role == "assistant":
                blocks: List[Dict[str, Any]] = []
                content = msg.get("content")
                if content:
                    blocks.append({"text": str(content)})
                for tc in msg.get("tool_calls", []) or []:
                    function = tc.get("function", {}) or {}
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({
                        "toolUse": {
                            "toolUseId": tc.get("id", ""),
                            "name": function.get("name", ""),
                            "input": args or {},
                        }
                    })
                if blocks:
                    messages.append({"role": "assistant", "content": blocks})
                continue
            if role == "tool":
                tool_id = msg.get("tool_call_id")
                content = msg.get("content", "")
                payload = None
                if isinstance(content, str):
                    try:
                        payload = json.loads(content)
                    except json.JSONDecodeError:
                        payload = None
                tool_content: List[Dict[str, Any]] = []
                if payload is not None:
                    tool_content.append({"json": payload})
                else:
                    tool_content.append({"text": str(content)})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_id,
                                "content": tool_content,
                            }
                        }
                    ],
                })

        return system, messages

    async def _execute_tool_mcp(
        self,
        session: ClientSession,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute tool via MCP."""
        try:
            result = await session.call_tool(tool_name, args)

            if result.content:
                content = result.content[0]
                if hasattr(content, "text"):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"result": content.text}
                elif hasattr(content, "data"):
                    return content.data
            return {"result": "success"}
        except Exception as e:
            LOGGER.warning("MCP tool execution failed: %s", e)
            return {"error": str(e)}

    def _check_completion(self, tool_name: str, result: Dict[str, Any]) -> bool:
        """Check if result indicates task completion."""
        if tool_name == "list_actions":
            return False

        if result.get("terminated", False):
            return True

        info = result.get("info", {})
        if info.get("episode_done", False):
            return True

        return False

    def _appears_complete(self, content: Optional[str]) -> bool:
        """Heuristic check if model thinks task is done."""
        if not content:
            return False
        lower = content.lower()
        completion_phrases = [
            "task complete",
            "task is complete",
            "completed the task",
            "diagnosis submitted",
            "submitted my diagnosis",
            "finished",
        ]
        return any(phrase in lower for phrase in completion_phrases)
    
    def _truncate_result(self, result: Dict[str, Any], max_len: int = 500) -> Dict[str, Any]:
        """Truncate large result values for event streaming."""
        truncated = {}
        for key, value in result.items():
            if isinstance(value, str) and len(value) > max_len:
                truncated[key] = value[:max_len] + "..."
            elif isinstance(value, dict):
                truncated[key] = self._truncate_result(value, max_len)
            elif isinstance(value, list) and len(str(value)) > max_len:
                truncated[key] = f"[{len(value)} items]"
            else:
                truncated[key] = value
        return truncated


async def run_gpt_agent(
    mcp_url: str,
    task_hint: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    max_turns: int = 25,
    verbose: bool = False,
    on_event: Optional[callable] = None,
) -> Dict[str, Any]:
    """Run GPT agent."""
    agent = GPTAgent(mcp_url=mcp_url, max_turns=max_turns, verbose=verbose, on_event=on_event)
    return await agent.run(task_hint=task_hint, task_context=task_context)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT solver agent using MCP")
    parser.add_argument("--mcp-url", required=True, help="MCP server URL")
    parser.add_argument("--task-hint", default=None, help="Task hint from green agent")
    parser.add_argument("--max-turns", type=int, default=25, help="Max turns (default: 25)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = asyncio.run(
        run_gpt_agent(
            mcp_url=args.mcp_url,
            task_hint=args.task_hint,
            max_turns=args.max_turns,
            verbose=args.verbose,
        )
    )

    print("\n" + "=" * 50)
    print("AGENT SESSION COMPLETE")
    print("=" * 50)
    print(f"Turns taken: {result.get('turns', 0)}")
    print(f"Task completed: {result.get('task_completed', False)}")
    if result.get("max_turns_reached"):
        print("WARNING: Max turns reached")
    if result.get("error"):
        print(f"ERROR: {result['error']}")


if __name__ == "__main__":
    main()
