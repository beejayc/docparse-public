import base64

from docparse.vlm.prompts import GROUNDING_PROMPT, TRANSCRIBE_PROMPT
from docparse.vlm.request import parse, to_request


def test_to_request_grounding() -> None:
    b64 = base64.b64encode(b"png").decode()
    body = to_request({"png_b64": b64})
    parts = body["messages"][0]["content"]
    assert parts[0]["image_url"]["url"] == f"data:image/png;base64,{b64}"
    assert parts[1] == {"type": "text", "text": GROUNDING_PROMPT}
    assert body["temperature"] == 0.2 and body["max_tokens"] == 4096


def test_to_request_transcribe_has_only_image() -> None:
    body = to_request({"png_b64": "x"}, prompt=TRANSCRIBE_PROMPT)
    assert len(body["messages"][0]["content"]) == 1


def test_parse_stores_raw_unparsed() -> None:
    resp = {
        "model": "m",
        "choices": [{"message": {"content": "![text](1,2,3,4)\nhi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    assert parse({}, resp) == {
        "raw": "![text](1,2,3,4)\nhi",
        "finish_reason": "stop",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "model": "m",
    }
