from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from trading_algo.broker.base import Broker
from trading_algo.config import IBKRConfig, TradingConfig
from trading_algo.llm.chat_protocol import ChatModelReply, ToolCall, format_tool_result_for_model, parse_chat_model_reply
from trading_algo.llm.config import LLMConfig
from trading_algo.llm.gemini import GeminiClient, LLMClient
from trading_algo.llm.tools import ToolError, dispatch_tool, list_tools
from trading_algo.oms import OrderManager
from trading_algo.risk import RiskLimits, RiskManager


_COLOR_ENABLED = True


def _c(code: str) -> str:
    if not _COLOR_ENABLED:
        return ""
    return f"\033[{code}m"


def _reset() -> str:
    return _c("0")


def _banner(title: str) -> str:
    line = "=" * max(10, len(title) + 6)
    return f"{_c('1;36')}{line}\n== {title} ==\n{line}{_reset()}"


def _pp(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(obj)


@dataclass
class ChatSession:
    broker: Broker
    trading: TradingConfig
    llm: LLMConfig
    client: LLMClient
    risk: RiskManager
    confirm_token: str | None = None
    stream: bool = True
    show_raw: bool = False
    max_tool_rounds: int = 5

    def __post_init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "text": str(text)})

    def run_turn(
        self,
        *,
        on_stream_token: Callable[[str], None] | None = None,
        on_tool_executed: Callable[[ToolCall, bool, Any], None] | None = None,
    ) -> ChatModelReply:
        """
        Executes one assistant turn, including any tool calls (iterative).
        """
        if not self.llm.enabled or self.llm.provider != "gemini":
            raise RuntimeError("Chat requires LLM_ENABLED=true and LLM_PROVIDER=gemini")
        if not self.llm.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY must be set")
        if not self.llm.allowed_symbols():
            raise RuntimeError("LLM_ALLOWED_SYMBOLS must be set (comma-separated)")

        oms = OrderManager(self.broker, self.trading, confirm_token=self.confirm_token)
        try:
            last_reply: ChatModelReply | None = None
            for _round in range(int(self.max_tool_rounds)):
                try:
                    raw = self._call_model(on_stream_token=on_stream_token)
                except Exception as exc:
                    msg = f"LLM request failed: {exc}"
                    self._messages.append({"role": "assistant", "text": msg})
                    return ChatModelReply(assistant_message=msg, tool_calls=[])
                if self.show_raw:
                    self._messages.append({"role": "assistant", "text": raw})
                reply = parse_chat_model_reply(raw)
                last_reply = reply
                if reply.assistant_message:
                    self._messages.append({"role": "assistant", "text": reply.assistant_message})
                if not reply.tool_calls:
                    return reply

                for call in reply.tool_calls:
                    ok, result = self._execute_tool(call, oms)
                    if on_tool_executed is not None:
                        on_tool_executed(call, ok, result)
                    self._messages.append({"role": "user", "text": format_tool_result_for_model(call=call, ok=ok, result=result)})
            return last_reply or ChatModelReply(assistant_message="", tool_calls=[])
        finally:
            oms.close()

    def _call_model(self, *, on_stream_token: Callable[[str], None] | None) -> str:
        prompt = _build_prompt(self._messages, self.llm.allowed_symbols(), self.llm.allowed_kinds(), list_tools())
        use_search = bool(self.llm.gemini_use_google_search)
        if not self.stream:
            try:
                return self.client.generate(prompt=prompt, system=_SYSTEM_PROMPT, use_google_search=use_search)
            except Exception:
                # Grounding can be restricted; retry without it.
                if use_search:
                    return self.client.generate(prompt=prompt, system=_SYSTEM_PROMPT, use_google_search=False)
                raise

        buf: list[str] = []
        try:
            for chunk in self.client.stream_generate(prompt=prompt, system=_SYSTEM_PROMPT, use_google_search=use_search):
                buf.append(str(chunk))
                if on_stream_token is not None:
                    on_stream_token(str(chunk))
        except Exception:
            # Retry once without Google Search if grounding causes request errors.
            if use_search:
                buf = []
                for chunk in self.client.stream_generate(prompt=prompt, system=_SYSTEM_PROMPT, use_google_search=False):
                    buf.append(str(chunk))
                    if on_stream_token is not None:
                        on_stream_token(str(chunk))
            else:
                raise
        text = "".join(buf)
        if text.strip() == "":
            # If streaming yields nothing (e.g. SSE parsing differences), fall back to non-streaming.
            return self.client.generate(prompt=prompt, system=_SYSTEM_PROMPT, use_google_search=use_search)
        return text

    def _execute_tool(self, call: ToolCall, oms: OrderManager) -> tuple[bool, Any]:
        try:
            result = dispatch_tool(
                call_name=call.name,
                call_args=call.args,
                broker=self.broker,
                oms=oms,
                allowed_kinds=self.llm.allowed_kinds(),
                allowed_symbols=self.llm.allowed_symbols(),
            )
            return True, result
        except ToolError as exc:
            return False, {"error": str(exc)}
        except Exception as exc:
            return False, {"error": str(exc)}


_SYSTEM_PROMPT = (
    "You are a terminal trading assistant for a PAPER-trading only OMS.\n"
    "You MUST respond with a single JSON object (no markdown).\n"
    "Schema:\n"
    "{\n"
    '  "assistant_message": "text for the human",\n'
    '  "tool_calls": [\n'
    '    {"id":"optional","name":"tool_name","args":{...}}\n'
    "  ]\n"
    "}\n"
    "If you do not need tools, return tool_calls=[].\n"
    "Never place orders for symbols outside allowed_symbols.\n"
)


def _build_prompt(
    messages: list[dict[str, str]],
    allowed_symbols: set[str],
    allowed_kinds: set[str],
    tools: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "now_epoch_s": time.time(),
            "allowed_symbols": sorted([s for s in allowed_symbols]),
            "allowed_kinds": sorted([k for k in allowed_kinds]),
            "tools": tools,
            "messages": list(messages),
        },
        sort_keys=True,
    )


def _load_dotenv_if_present() -> None:
    if not os.path.exists(".env"):
        return
    with open(".env", "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading_algo.llm.chat", description="Interactive terminal chat (Gemini + OMS tools)")
    p.add_argument("--broker", choices=["ibkr", "sim"], default=None, help="Override TRADING_BROKER")
    p.add_argument("--confirm-token", default=None, help="Must match TRADING_ORDER_TOKEN to send IBKR orders")
    p.add_argument("--ibkr-host", default=None)
    p.add_argument("--ibkr-port", default=None)
    p.add_argument("--ibkr-client-id", default=None)
    p.add_argument("--no-stream", action="store_true", help="Disable Gemini streaming")
    p.add_argument("--show-raw", action="store_true", help="Also store/display raw model JSON")
    p.add_argument("--no-color", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    global _COLOR_ENABLED
    _load_dotenv_if_present()
    cfg = TradingConfig.from_env()
    llm_cfg = LLMConfig.from_env()
    args = build_parser().parse_args(argv)

    if args.no_color:
        _COLOR_ENABLED = False

    ibkr = IBKRConfig(
        host=args.ibkr_host or cfg.ibkr.host,
        port=int(args.ibkr_port or cfg.ibkr.port),
        client_id=int(args.ibkr_client_id or cfg.ibkr.client_id),
    )
    cfg = TradingConfig(
        broker=args.broker or cfg.broker,
        live_enabled=cfg.live_enabled,
        require_paper=True,
        dry_run=cfg.dry_run,
        order_token=cfg.order_token,
        db_path=cfg.db_path,
        poll_seconds=cfg.poll_seconds,
        ibkr=ibkr,
    )

    if not str(llm_cfg.gemini_model).startswith("gemini-3"):
        raise SystemExit(f"Refusing to run with GEMINI_MODEL={llm_cfg.gemini_model!r}; set GEMINI_MODEL=gemini-3")
    if not llm_cfg.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is empty; set it in .env or your shell to use chat.")

    if cfg.broker == "sim":
        from trading_algo.broker.sim import SimBroker

        broker: Broker = SimBroker()
        # Provide one default quote so "get_snapshot" works immediately.
        from trading_algo.instruments import InstrumentSpec

        broker.connect()
        broker.set_market_data(InstrumentSpec(kind="STK", symbol="AAPL"), last=100.0)  # type: ignore[attr-defined]
        broker.disconnect()
    else:
        from trading_algo.broker.ibkr import IBKRBroker

        broker = IBKRBroker(cfg.ibkr, require_paper=True)

    client = GeminiClient(api_key=llm_cfg.gemini_api_key or "", model=llm_cfg.gemini_model)

    print(_banner("IBKR Paper OMS Chat"))
    print(f"broker={cfg.broker} dry_run={cfg.dry_run} live_enabled={cfg.live_enabled} db={cfg.db_path or 'off'}")
    print(f"allowed_symbols={','.join(sorted(llm_cfg.allowed_symbols())) or '(missing)'} model={llm_cfg.gemini_model} stream={not args.no_stream}")
    print("Type /help for commands.")

    broker.connect()
    try:
        session = ChatSession(
            broker=broker,
            trading=cfg,
            llm=llm_cfg,
            client=client,
            risk=RiskManager(RiskLimits()),
            confirm_token=args.confirm_token,
            stream=not bool(args.no_stream),
            show_raw=bool(args.show_raw),
        )

        while True:
            try:
                user = input(f"{_c('1;32')}you>{_reset()} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not user:
                continue
            if user in {"/quit", "/exit"}:
                return 0
            if user == "/help":
                print(_banner("Commands"))
                print("/help  /exit  /quit")
                print("The assistant can call tools: get_snapshot, get_positions, get_account, list_open_orders, place_order, modify_order, cancel_order, oms_reconcile, oms_track.")
                continue

            session.add_user_message(user)

            print(f"{_c('1;34')}assistant>{_reset()} ", end="", flush=True)

            # In streaming mode the model output is structured JSON; we don't print raw chunks.
            # Instead, show a lightweight progress indicator while we stream into a buffer.
            stream_chunks = 0

            def _on_token(_tok: str) -> None:
                nonlocal stream_chunks
                stream_chunks += 1
                if stream_chunks in {1, 25, 50, 100, 200, 400}:
                    sys.stdout.write(".")
                    sys.stdout.flush()

            executed: list[tuple[ToolCall, bool, Any]] = []

            def _on_tool(call: ToolCall, ok: bool, result: Any) -> None:
                executed.append((call, ok, result))

            try:
                reply = session.run_turn(
                    on_stream_token=_on_token if session.stream else None,
                    on_tool_executed=_on_tool,
                )
            except Exception as exc:
                # Never crash the UI; keep the broker connected and continue.
                if session.stream:
                    print()
                print(f"{_c('1;31')}error:{_reset()} {exc}")
                continue
            if session.stream:
                print()

            # If we didn't stream human-readable text (we only show dots), print the final message now.
            if reply.assistant_message.strip():
                print(reply.assistant_message)
            else:
                print("(no assistant_message returned)")

            if executed:
                print(_banner("Tool Calls"))
                for call, ok, result in executed:
                    status = "ok" if ok else "error"
                    color = "1;32" if ok else "1;31"
                    print(f"{_c('1;33')}{call.name}{_reset()} args={_pp(call.args)}")
                    print(f"  -> {_c(color)}{status}{_reset()} result={_pp(result)}")
                print(_banner("Note"))
                print("Tool results are automatically fed back into the model as the next message.")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
