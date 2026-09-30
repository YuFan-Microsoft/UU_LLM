"""
Tests for the local-server HTTP client, run against a real in-process HTTP
server that speaks the chat-completions API (no network, no model needed).

    python tests/test_llm_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.io_utils import read_jsonl  # noqa: E402
from modules.llm_client import (  # noqa: E402
    LLMHTTPError,
    LocalLLMClient,
    invoke_chat,
    normalize_base_url,
    parse_json_result,
    resolve_model,
)
from test_pipeline_offline import FakeLLM  # noqa: E402

import run as run_cli  # noqa: E402


class LocalServer:
    """Threaded HTTP server imitating vLLM/Ollama: ``GET /v1/models``, ``POST /v1/chat/completions``.

    ``fail_first`` makes the first N chat requests return ``fail_status``.
    ``reply`` returns the assistant content for a request body (default: FakeLLM pipeline replies).
    """

    def __init__(self, models=("local-model",), fail_first=0, fail_status=503, reply=None):
        self.requests = []
        self.models = list(models)
        self.fail_left = fail_first
        self.fail_status = fail_status
        self.reply = reply
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in server.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": "not found"})
                if server.fail_left > 0:
                    server.fail_left -= 1
                    return self._send(server.fail_status, {"error": "busy"})
                if server.reply is not None:
                    content = server.reply(body)
                    return self._send(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                                         "message": {"role": "assistant", "content": content}}],
                                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
                self._send(200, FakeLLM.respond(body))

        ThreadingHTTPServer.request_queue_size = 128  # default 5 resets connections under concurrency
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_normalize_base_url():
    assert normalize_base_url("http://localhost:8000") == "http://localhost:8000/v1"
    assert normalize_base_url("http://localhost:8000/v1/") == "http://localhost:8000/v1"
    assert normalize_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert normalize_base_url("http://h/custom/v1/chat/completions") == "http://h/custom/v1"
    assert normalize_base_url("http://h/serve/chat/completions") == "http://h/serve"


def test_request_body_and_no_auth_header():
    with LocalServer(reply=lambda body: '{"ok": true}') as srv:
        client = LocalLLMClient(srv.url)
        text, usage, _, _ = asyncio.run(invoke_chat(
            client, "local-model", [{"role": "user", "content": "hi"}], 16000,
            seed=7, max_output_tokens=4096, disable_thinking=True))
        assert json.loads(text) == {"ok": True} and usage["total_tokens"] == 2
        req = srv.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert {k.lower() for k in req["headers"]}.isdisjoint({"authorization", "api-key"})
        assert req["body"] == {
            "model": "local-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "temperature": 0.2,
            "seed": 7,
            "chat_template_kwargs": {"enable_thinking": False},
        }


def test_model_auto_detect():
    with LocalServer(models=["qwen3-8b", "other"]) as srv:
        client = LocalLLMClient(srv.url)
        assert asyncio.run(resolve_model(client, "")) == "qwen3-8b"
        assert asyncio.run(resolve_model(client, "explicit")) == "explicit"


def test_retry_transient_then_succeed_and_no_retry_on_client_error():
    with LocalServer(fail_first=2, fail_status=503, reply=lambda b: "{}") as srv:
        client = LocalLLMClient(srv.url, max_retries=2, retry_delay=0.01)
        asyncio.run(invoke_chat(client, "m", [{"role": "user", "content": "x"}], 100))
        assert len(srv.requests) == 3

    with LocalServer(fail_first=5, fail_status=400, reply=lambda b: "{}") as srv:
        client = LocalLLMClient(srv.url, max_retries=2, retry_delay=0.01)
        try:
            asyncio.run(invoke_chat(client, "m", [{"role": "user", "content": "x"}], 100))
            raise AssertionError("expected LLMHTTPError")
        except LLMHTTPError as e:
            assert e.status == 400
        assert len(srv.requests) == 1


def test_parse_fences_and_truncation():
    assert parse_json_result('```json\n{"a": [1, 2]}\n```').value == {"a": [1, 2]}
    assert parse_json_result('{"a": [1, 2]').value == {"a": [1, 2]}


def test_think_output_is_an_error_like_production():
    think = "<think>hmm</think>{\"ok\": 1}"
    with LocalServer(reply=lambda body: think) as srv:
        client = LocalLLMClient(srv.url)
        try:
            asyncio.run(invoke_chat(client, "m", [{"role": "user", "content": "x"}], 100))
            raise AssertionError("expected AssertionError for <think> output")
        except AssertionError as e:
            assert "<think>" in str(e)
        # --allow-thinking (non-production) strips it instead.
        text, *_ = asyncio.run(invoke_chat(client, "m", [{"role": "user", "content": "x"}], 100,
                                           disable_thinking=False))
        assert json.loads(text) == {"ok": 1}
        assert "chat_template_kwargs" not in srv.requests[-1]["body"]


def test_default_request_matches_production_gemma4():
    with LocalServer(reply=lambda body: "{}") as srv:
        client = LocalLLMClient(srv.url)
        asyncio.run(invoke_chat(client, "gemma4", [{"role": "user", "content": "x"}], 32000,
                                max_output_tokens=8192))
        assert srv.requests[0]["body"] == {
            "messages": [{"role": "user", "content": "x"}], "model": "gemma4",
            "max_tokens": 8192, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }


def test_cli_end_to_end_over_http():
    """run.py → real HTTP → local server, model auto-detected, full layer1→layer3."""
    with LocalServer(models=["my-local-model"]) as srv, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        code = run_cli.main(["--input", str(ROOT / "sample_data" / "sample_input.jsonl"), "--output", str(out),
                             "--url", srv.url, "--workers", "4"])
        assert code == 0
        summary = json.loads((out / "run_summary.json").read_text())
        assert summary["model"] == "my-local-model" and summary["failed_users"] == []
        assert {r["user_id"] for r in read_jsonl(out / "final_profiles.jsonl")} == {"user_a", "user_b"}
        assert srv.requests and all(r["body"]["model"] == "my-local-model" for r in srv.requests)


if __name__ == "__main__":
    test_normalize_base_url()
    test_request_body_and_no_auth_header()
    test_model_auto_detect()
    test_retry_transient_then_succeed_and_no_retry_on_client_error()
    test_parse_fences_and_truncation()
    test_think_output_is_an_error_like_production()
    test_default_request_matches_production_gemma4()
    test_cli_end_to_end_over_http()
    print("OK")
