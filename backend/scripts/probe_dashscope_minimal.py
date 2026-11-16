"""Minimal DashScope connectivity probe. No real notes. No key printing."""
import json
import sys

import httpx

from douyin_library.config import Settings

s = Settings()
base = (s.note_external_base_url or "").rstrip("/")
model = s.note_external_model
key = s.note_external_api_key
if not base or not model or not key:
    print("CONFIG: incomplete")
    sys.exit(2)

url = f"{base}/chat/completions"
headers = {"Authorization": f"Bearer {key.get_secret_value().strip()}"}
body = {
    "model": model,
    "messages": [{"role": "user", "content": "请只回复两个字：正常"}],
    "max_tokens": 16,
}

try:
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=body)
    print("HTTP_STATUS:", resp.status_code)
    print("CONTENT_TYPE:", resp.headers.get("content-type", "?"))
    try:
        payload = resp.json()
        print("JSON_PARSE: ok")
        print("RESP_MODEL:", payload.get("model", "<absent>"))
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            print("CONTENT_KIND:", type(content).__name__)
            if isinstance(content, str):
                print("CONTENT_SNIPPET:", content[:120])
            else:
                print("CONTENT_SNIPPET:", str(content)[:120])
        elif isinstance(payload.get("error"), dict):
            err = payload["error"]
            print("ERROR_CODE:", err.get("code"))
            print("ERROR_MESSAGE:", str(err.get("message"))[:300])
        else:
            print("PAYLOAD_KEYS:", sorted(payload.keys())[:12])
    except ValueError:
        print("JSON_PARSE: failed")
        print("BODY_SNIPPET:", resp.text[:200])
except Exception as exc:  # noqa: BLE001
    print("TRANSPORT_ERROR:", type(exc).__name__, str(exc)[:200])
