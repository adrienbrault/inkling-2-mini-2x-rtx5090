#!/usr/bin/env python3
"""Served-path smoke: tool call, image input, temperature > 0 with the drafter (4 requests). Records HTTP status and
error bodies. Set URL to the server root and SMOKE_MODEL to the served name.
Usage: smoke_features.py OUT.json"""
import base64, json, os, struct, sys, urllib.error, urllib.request, zlib
URL = (os.environ.get("URL") or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}").rstrip("/")
CHAT_URL = URL + "/v1/chat/completions"

def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

def call(body):
    req = urllib.request.Request(CHAT_URL, json.dumps({"model": os.environ.get("SMOKE_MODEL", "inkling-2-mini"), **body}).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:600]

out = {}
code, r = call({"messages": [{"role": "user", "content": "What is the weather in Paris right now?"}], "temperature": 0,
                "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
                "tool_choice": "auto", "max_tokens": 1024})
tc = r["choices"][0]["message"].get("tool_calls") if code == 200 else None
out["tool_call"] = {"status": code, "ok": bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Paris" in tc[0]["function"]["arguments"],
                    "detail": tc if tc else (r if code != 200 else r["choices"][0]["message"])}
img = "data:image/png;base64," + base64.b64encode(png(64, 64, (220, 20, 20))).decode()
code, r = call({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "What single color fills this image? One word."}]}], "temperature": 0, "max_tokens": 512,
                "chat_template_kwargs": {"reasoning_effort": "none"}})
txt = r["choices"][0]["message"].get("content") if code == 200 else None
out["image"] = {"status": code, "ok": bool(txt) and "red" in txt.lower(), "detail": txt if code == 200 else r}
out["t08"] = []
for i in range(4):
    code, r = call({"messages": [{"role": "user", "content": "Write a short poem about GPUs, then a haiku."}], "temperature": 0.8,
                    "max_tokens": 1200, "chat_template_kwargs": {"reasoning_effort": "none"}})
    if code == 200:
        m = r["choices"][0]["message"]
        out["t08"].append({"status": 200, "ok": bool((m.get("content") or "").strip()), "tokens": r["usage"]["completion_tokens"],
                           "finish": r["choices"][0]["finish_reason"], "head": (m.get("content") or "")[:60]})
    else:
        out["t08"].append({"status": code, "ok": False, "detail": r})
out["all_ok"] = out["tool_call"]["ok"] and out["image"]["ok"] and all(x["ok"] for x in out["t08"])
with open(sys.argv[1], "w") as report:
    json.dump(out, report, indent=1, ensure_ascii=False)
print(json.dumps({"tool_call": out["tool_call"]["ok"], "image": out["image"]["ok"], "t08": [x["ok"] for x in out["t08"]],
                  "all_ok": out["all_ok"]}))
if not out["all_ok"]:
    failures = [x for x in (out["tool_call"], out["image"], *out["t08"]) if not x["ok"]]
    print(json.dumps(failures), file=sys.stderr)
    sys.exit(1)
