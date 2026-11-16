"""Probe whether the configured model honors response_format=json_schema strict."""
import httpx

from douyin_library.config import Settings

s = Settings()
base = (s.note_external_base_url or "").rstrip("/")
model = s.note_external_model
key = s.note_external_api_key

url = f"{base}/chat/completions"
headers = {"Authorization": f"Bearer {key.get_secret_value().strip()}"}
body = {
    "model": model,
    "messages": [{"role": "user", "content": '输出 {"ok": "yes"}'}],
    "max_tokens": 64,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "probe_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "string"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    },
}

try:
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=body)
    print("HTTP_STATUS:", resp.status_code)
    try:
        payload = resp.json()
        print("JSON_PARSE: ok")
        print("RESP_MODEL:", payload.get("model", "<absent>"))
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            content = (choices[0].get("message") or {}).get("content")
            print("CONTENT:", str(content)[:200])
        elif isinstance(payload.get("error"), dict):
            err = payload["error"]
            print("ERROR_CODE:", err.get("code"))
            print("ERROR_MESSAGE:", str(err.get("message"))[:300])
    except ValueError:
        print("JSON_PARSE: failed")
        print("BODY_SNIPPET:", resp.text[:200])
except Exception as exc:  # noqa: BLE001
    print("TRANSPORT_ERROR:", type(exc).__name__, str(exc)[:200])
