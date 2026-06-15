from __future__ import annotations

import argparse
import getpass
import json
import os
import time
from pprint import pprint
from typing import Any

from veilrouter import Router, RouterConfig
from veilrouter.pii.detector import RegexPiiDetector
from veilrouter.providers.base import ChatResponse, Message
from veilrouter.providers.openai_compatible import OpenAICompatibleProvider

PROMPT = "Send a renewal-risk summary to Jane Doe at jane@example.com and call +1 (425) 555-0199 if blocked."


class FixedScorer:
    def __init__(self, score: int, *, fail_if_called: bool = False) -> None:
        self.value = score
        self.fail_if_called = fail_if_called
        self.calls = 0

    def score(self, text: str) -> int:
        self.calls += 1
        if self.fail_if_called:
            raise AssertionError("scorer should not be called")
        return self.value


class RecordingProvider:
    def __init__(self, model: str, response: ChatResponse | None = None) -> None:
        self.model = model
        self.response = response or ChatResponse(text="ok", model=model, tokens_in=1, tokens_out=1)
        self.messages: list[Message] | None = None
        self.opts: dict[str, Any] | None = None

    def complete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        self.messages = messages
        self.opts = opts
        return self.response

    def stream(self, messages: list[Message], **opts: Any):
        raise NotImplementedError("streaming is not used in this diagnostic")


class ForwardingRecordingProvider:
    def __init__(self, provider: OpenAICompatibleProvider) -> None:
        self.provider = provider
        self.model = provider.model
        self.messages: list[Message] | None = None
        self.opts: dict[str, Any] | None = None

    def complete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        self.messages = messages
        self.opts = opts
        return self.provider.complete(messages, **opts)

    def stream(self, messages: list[Message], **opts: Any):
        return self.provider.stream(messages, **opts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify veilrouter local/cloud privacy behavior.")
    parser.add_argument("--live-openrouter", action="store_true", help="Also call OpenRouter openrouter/free.")
    parser.add_argument("--cloud-model", default="openrouter/free", help="OpenRouter model slug for --live-openrouter.")
    parser.add_argument("--max-tokens", type=int, default=96, help="Max tokens for the optional live OpenRouter call.")
    args = parser.parse_args()

    run_local_route_demo()
    run_cloud_text_redaction_demo()
    run_cloud_tool_call_demo()
    if args.live_openrouter:
        run_live_openrouter_demo(args.cloud_model, args.max_tokens)
    print_section("done")
    print("All selected diagnostics passed.")
    return 0


def run_local_route_demo() -> None:
    print_section("1. local route keeps PII local")
    local = RecordingProvider("fake-local", ChatResponse(text="Local handled raw request.", model="fake-local"))
    cloud = RecordingProvider("fake-cloud")
    router = Router(
        RouterConfig(pii_regex_backstop=True),
        local_provider=local,
        cloud_provider=cloud,
        scorer=FixedScorer(0),
        pii_detector=RegexPiiDetector(),
    )

    response = router.run(PROMPT)
    local_seen = first_content(local.messages)

    print("original prompt:")
    print(PROMPT)
    print("local provider received:")
    print(local_seen)
    print("response route:", response.route)
    print("pii_detected:", response.pii_detected)

    assert response.route == "local"
    assert "Jane Doe" in local_seen
    assert "jane@example.com" in local_seen
    assert "+1 (425) 555-0199" in local_seen
    assert cloud.messages is None
    assert response.pii_detected is False


def run_cloud_text_redaction_demo() -> None:
    print_section("2. cloud route redacts before send and restores text response")
    cloud = RecordingProvider(
        "fake-cloud",
        ChatResponse(text="Cloud saw [PERSON_NAME_1], [EMAIL_1], and [PHONE_1].", model="fake-cloud", tokens_in=10, tokens_out=4),
    )
    router = Router(
        RouterConfig(cloud_model="fake-cloud", pii_regex_backstop=True),
        local_provider=RecordingProvider("fake-local"),
        cloud_provider=cloud,
        scorer=FixedScorer(5),
        pii_detector=RegexPiiDetector(),
    )

    response = router.run(PROMPT)
    cloud_seen = first_content(cloud.messages)

    print("original prompt:")
    print(PROMPT)
    print("cloud provider received:")
    print(cloud_seen)
    print("restored response text:")
    print(response.text)
    print("redaction metadata:", response.redaction_count, response.redaction_categories)

    assert response.route == "cloud"
    assert "Jane Doe" not in cloud_seen
    assert "jane@example.com" not in cloud_seen
    assert "+1 (425) 555-0199" not in cloud_seen
    assert "[PERSON_NAME_1]" in cloud_seen
    assert "[EMAIL_1]" in cloud_seen
    assert "[PHONE_1]" in cloud_seen
    assert "Jane Doe" in response.text
    assert "jane@example.com" in response.text
    assert "+1 (425) 555-0199" in response.text
    assert response.redaction_categories == {"PERSON_NAME": 1, "EMAIL": 1, "PHONE": 1}


def run_cloud_tool_call_demo() -> None:
    print_section("3. tool calls force cloud and restore tool-call arguments")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "send_followup",
                "arguments": '{"name": "[PERSON_NAME_1]", "email": "[EMAIL_1]", "phone": "[PHONE_1]"}',
            },
        }
    ]
    cloud = RecordingProvider("fake-cloud", ChatResponse(text="", model="fake-cloud", tool_calls=tool_calls))
    scorer = FixedScorer(0, fail_if_called=True)
    router = Router(
        RouterConfig(local_score_max=5, cloud_model="fake-cloud", pii_regex_backstop=True),
        local_provider=RecordingProvider("fake-local"),
        cloud_provider=cloud,
        scorer=scorer,
        pii_detector=RegexPiiDetector(),
    )

    response = router.run(
        PROMPT,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "send_followup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    cloud_seen = first_content(cloud.messages)

    print("cloud provider received:")
    print(cloud_seen)
    print("restored tool calls:")
    pprint(response.tool_calls)
    print("local tool execution with restored arguments:")
    tool_result = execute_restored_tool_call(response.tool_calls[0])
    pprint(tool_result)
    print("score:", response.score, "(None means scorer was skipped because tools require cloud)")

    assert response.route == "cloud"
    assert response.score is None
    assert scorer.calls == 0
    assert "Jane Doe" not in cloud_seen
    assert "jane@example.com" not in cloud_seen
    assert "+1 (425) 555-0199" not in cloud_seen
    assert "[PERSON_NAME_1]" in cloud_seen
    assert response.tool_calls[0]["function"]["arguments"] == '{"name": "Jane Doe", "email": "jane@example.com", "phone": "+1 (425) 555-0199"}'
    assert tool_result == {
        "sent": True,
        "recipient": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1 (425) 555-0199",
    }


def run_live_openrouter_demo(model: str, max_tokens: int) -> None:
    print_section("4. live OpenRouter cloud call through veilrouter")
    api_key = os.getenv("VEILROUTER_CLOUD_API_KEY") or getpass.getpass("OpenRouter API key: ")
    provider = ForwardingRecordingProvider(
        OpenAICompatibleProvider(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    )
    router = Router(
        RouterConfig(cloud_model=model, pii_regex_backstop=True),
        local_provider=RecordingProvider("fake-local"),
        cloud_provider=provider,
        scorer=FixedScorer(5),
        pii_detector=RegexPiiDetector(),
    )

    print("calling OpenRouter...", flush=True)
    started = time.perf_counter()
    response = router.run(
        PROMPT,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "https://github.com/MaanavD/veilrouter",
            "X-OpenRouter-Title": "veilrouter-privacy-diagnostic",
        },
    )
    elapsed = time.perf_counter() - started
    cloud_seen = first_content(provider.messages)

    print(f"completed in {elapsed:.2f}s")
    print("cloud provider received:")
    print(cloud_seen)
    print("live response text:")
    print(response.text)
    print("metadata:")
    print(
        {
            "route": response.route,
            "score": response.score,
            "model": response.model,
            "pii_detected": response.pii_detected,
            "redaction_count": response.redaction_count,
            "redaction_categories": response.redaction_categories,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost_estimate": response.cost_estimate,
        }
    )

    assert response.route == "cloud"
    assert "Jane Doe" not in cloud_seen
    assert "jane@example.com" not in cloud_seen
    assert "+1 (425) 555-0199" not in cloud_seen
    assert "[PERSON_NAME_1]" in cloud_seen
    assert "[EMAIL_1]" in cloud_seen
    assert "[PHONE_1]" in cloud_seen


def execute_restored_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    if tool_call["function"]["name"] != "send_followup":
        raise ValueError(f"unexpected tool call: {tool_call['function']['name']}")
    args = json.loads(tool_call["function"]["arguments"])
    return send_followup(**args)


def send_followup(*, name: str, email: str, phone: str) -> dict[str, Any]:
    return {
        "sent": True,
        "recipient": name,
        "email": email,
        "phone": phone,
    }


def first_content(messages: list[Message] | None) -> str:
    assert messages, "provider was not called"
    content = messages[0]["content"]
    assert isinstance(content, str)
    return content


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


if __name__ == "__main__":
    raise SystemExit(main())
