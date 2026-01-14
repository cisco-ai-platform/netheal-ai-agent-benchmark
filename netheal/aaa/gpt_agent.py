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
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AzureOpenAI

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


class GPTAgent:
    """GPT-based solver using MCP and Azure OpenAI."""

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

        env_path = Path(__file__).parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            LOGGER.info("Loaded environment from %s", env_path)
        else:
            load_dotenv()

        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.mcp_tools: Dict[str, Any] = {}
        self.turn_count = 0
        self.task_completed = False
    
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
        LOGGER.info("  Max turns: %d", self.max_turns)

        try:
            async with streamablehttp_client(self.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    LOGGER.info("MCP session initialized")

                    tools_result = await session.list_tools()
                    self.tools = []
                    for tool in tools_result.tools:
                        self.mcp_tools[tool.name] = tool
                        openai_func = _mcp_tool_to_openai_function(tool)
                        self.tools.append(openai_func)
                        LOGGER.info("  Discovered tool: %s", tool.name)

                    if not self.tools:
                        LOGGER.error("No tools discovered")
                        return {"error": "No tools available", "turns": 0}

                    LOGGER.info("Discovered %d tools via MCP", len(self.tools))

                    return await self._run_loop(session, task_hint, task_context)

        except Exception as e:
            LOGGER.error("MCP session failed: %s", e)
            return {"error": str(e), "turns": self.turn_count}

    async def _run_loop(
        self,
        session: ClientSession,
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

            response = self._call_gpt()
            assistant_message = response.choices[0].message
            self.messages.append(assistant_message.model_dump())
            
            # Extract usage info for reasoning models
            usage_info = {}
            if response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    details = response.usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                        usage_info["reasoning_tokens"] = details.reasoning_tokens
            
            # Emit LLM response event with content/reasoning
            # Always emit, even if content is empty (model may proceed directly to tool calls)
            llm_content = assistant_message.content or ""
            self._emit_event("llm_response", {
                "content": llm_content,
                "has_tool_calls": bool(assistant_message.tool_calls),
                "finish_reason": response.choices[0].finish_reason,
                "model": response.model,
                "usage": usage_info,
                "raw_content_is_none": assistant_message.content is None,
                "raw_content_is_empty": assistant_message.content == "",
            })

            if assistant_message.tool_calls:
                # Emit tool calls event
                tool_call_summaries = []
                for tc in assistant_message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_call_summaries.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
                
                self._emit_event("tool_calls", {
                    "tools": tool_call_summaries,
                    "reasoning": llm_content,  # Include reasoning that led to tool calls
                })
                
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

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
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    })

                self.messages.extend(tool_results)
            else:
                if assistant_message.content:
                    LOGGER.info("  Assistant: %s", assistant_message.content[:200])
                    self._emit_event("assistant_message", {
                        "content": assistant_message.content,
                    })

                if self._appears_complete(assistant_message.content):
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

    def _call_gpt(self) -> Any:
        """Make chat completion call with reasoning support."""
        # Build extra parameters for reasoning models (GPT-5, o3, o4-mini)
        extra_params = {}
        if self.reasoning_effort:
            extra_params["reasoning_effort"] = self.reasoning_effort
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=self.messages,
            tools=self.tools if self.tools else None,
            tool_choice="auto" if self.tools else None,
            extra_body=extra_params if extra_params else None,
        )
        
        # Log full response for debugging (especially for reasoning models)
        LOGGER.info("GPT Response - Model: %s", response.model)
        LOGGER.info("GPT Response - Finish reason: %s", response.choices[0].finish_reason)
        LOGGER.info("GPT Response - Content: %s", response.choices[0].message.content)
        LOGGER.info("GPT Response - Tool calls: %s", 
                    len(response.choices[0].message.tool_calls) if response.choices[0].message.tool_calls else 0)
        
        # Check for reasoning tokens (GPT-5, o3, o4-mini reasoning models)
        usage = response.usage
        if usage:
            LOGGER.info("GPT Usage - Prompt tokens: %d, Completion tokens: %d", 
                        usage.prompt_tokens, usage.completion_tokens)
            # Reasoning models include reasoning_tokens in completion_tokens_details
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                reasoning_tokens = getattr(details, 'reasoning_tokens', None)
                if reasoning_tokens:
                    LOGGER.info("GPT Usage - Reasoning tokens: %d (reasoning_effort=%s)", 
                                reasoning_tokens, self.reasoning_effort)
                    # Emit reasoning tokens info
                    self._emit_event("reasoning_usage", {
                        "reasoning_tokens": reasoning_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "prompt_tokens": usage.prompt_tokens,
                        "reasoning_effort": self.reasoning_effort,
                    })
        
        return response

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
