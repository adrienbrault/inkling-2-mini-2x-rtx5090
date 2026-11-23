#!/usr/bin/env python3
"""Structured output and tool-choice smoke on a served boot (default :8000). For each model name given, both streaming
and not: response_format json_object and json_schema (strict), structured_outputs choice and regex, tool_choice
"required" and a named function, tool_choice "auto" with a tool. Checks each output parses / matches, then runs 8
schema requests concurrently beside 8 free-text requests (grammar rows mixed with plain rows in batch-sharded
sampling). Prints one JSON line per case and a summary.

  python3 structured_smoke.py OUT.json [--url http://127.0.0.1:8000] [--names inkling-2-mini,inkling-2-mini-ablit]
"""
import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
          "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4}},
          "required": ["city", "population", "landmarks"], "additionalProperties": False}
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city", "unit"]}}}
Q = "Give facts about Paris."
REQUEST_SETTINGS = {"max_tokens": 4096, "chat_template_kwargs": {"reasoning_effort": "none"}}


def call(url, body, stream):
    body = {**body, **REQUEST_SETTINGS, "stream": stream}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            d = json.load(r); m = d["choices"][0]["message"]
            return m.get("content") or "", m.get("tool_calls") or [], d["choices"][0]["finish_reason"], time.time() - t
        content, calls, fin = "", {}, None
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if "error" in d:
                raise RuntimeError(d["error"])
            if not d.get("choices"):
                continue
            c = d["choices"][0]; dl = c.get("delta", {}); fin = c.get("finish_reason") or fin
            content += dl.get("content") or ""
            for tc in dl.get("tool_calls") or []:
                e = calls.setdefault(tc.get("index", 0), {"function": {"name": "", "arguments": ""}})
                f = tc.get("function") or {}
                e["function"]["name"] += f.get("name") or ""; e["function"]["arguments"] += f.get("arguments") or ""
        return content, list(calls.values()), fin, time.time() - t


def check_schema(s):
    d = json.loads(s)
    return (
        isinstance(d, dict)
        and set(d) == {"city", "population", "landmarks"}
        and isinstance(d["city"], str)
        and type(d["population"]) is int
        and isinstance(d["landmarks"], list)
        and 2 <= len(d["landmarks"]) <= 4
        and all(isinstance(landmark, str) for landmark in d["landmarks"])
    )



def cases(name):
    base = {"model": name, "messages": [{"role": "user", "content": Q}]}
    return {
        "json_object": ({**base, "messages": [{"role": "user", "content": Q + " Answer in JSON."}], "response_format": {"type": "json_object"}},
                        lambda c, t: isinstance(json.loads(c), dict)),
        "json_schema": ({**base, "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "schema": SCHEMA, "strict": True}}},
                        lambda c, t: check_schema(c)),
        "choice": ({**base, "messages": [{"role": "user", "content": "Is Paris in France? Answer yes or no."}],
                    "structured_outputs": {"choice": ["yes", "no"]}}, lambda c, t: c.strip() in ("yes", "no")),
        "regex": ({**base, "messages": [{"role": "user", "content": "Write a date in 2026."}],
                   "structured_outputs": {"regex": r"2026-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9])"}},
                  lambda c, t: re.fullmatch(r"2026-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9])", c.strip()) is not None),
        "tool_required": ({**base, "messages": [{"role": "user", "content": "What's the weather in Lyon?"}], "tools": [TOOL], "tool_choice": "required"},
                          lambda c, t: t and json.loads(t[0]["function"]["arguments"])["city"]),
        "tool_named": ({**base, "messages": [{"role": "user", "content": "Weather in Nice please."}], "tools": [TOOL],
                        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
                       lambda c, t: t and t[0]["function"]["name"] == "get_weather" and json.loads(t[0]["function"]["arguments"])["unit"] in ("c", "f")),
        "tool_auto": ({**base, "messages": [{"role": "user", "content": "What's the weather in Marseille in celsius?"}], "tools": [TOOL]},
                      lambda c, t: t and t[0]["function"]["name"] == "get_weather"),
    }


def mixed_batch(url, name):
    schema = cases(name)["json_schema"][0]
    plain = {"model": name, "messages": [{"role": "user", "content": "Write two sentences about Paris."}]}

    def run(i):
        try:
            content, _, _, _ = call(url, schema if i % 2 == 0 else plain, False)
            return {"ok": check_schema(content) if i % 2 == 0 else bool(content.strip())}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(run, range(16)))
    return {"schema_rows_valid": sum(r["ok"] for r in rows[::2]), "of": 8,
            "plain_rows_nonempty": sum(r["ok"] for r in rows[1::2]),
            "errors": [r["error"] for r in rows if "error" in r]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default=os.environ.get("URL") or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")
    ap.add_argument("--names", help="Comma-separated served names. Defaults to every name in /v1/models.")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    if args.names:
        names = args.names.split(",")
    else:
        with urllib.request.urlopen(url + "/v1/models", timeout=60) as response:
            names = [m["id"] for m in json.load(response)["data"]]
    if not names or any(not name for name in names):
        ap.error("no served names to check")

    results, mixed = [], {}
    for name in names:
        for case, (body, ok) in cases(name).items():
            for stream in (False, True):
                result = {"name": name, "case": case, "stream": stream}
                try:
                    c, t, fin, s = call(url, body, stream)
                    result.update(ok=bool(ok(c, t)), finish=fin, s=round(s, 2), content=c[:200], tool_calls=t[:1])
                except Exception as exc:
                    result.update(ok=False, error=str(exc))
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ("content", "tool_calls")}), flush=True)
        mixed[name] = mixed_batch(url, name)
        print("mixed", json.dumps({"name": name, **mixed[name]}), flush=True)

    failed = [(r["name"], r["case"], r["stream"]) for r in results if not r["ok"]]
    summary = {"passed": len(results) - len(failed), "of": len(results), "failed": failed, "mixed": mixed,
               "request_settings": REQUEST_SETTINGS}
    print("summary", json.dumps(summary), flush=True)
    with open(args.out, "w") as report:
        json.dump({"cases": results, "summary": summary}, report, indent=1, ensure_ascii=False)
    if failed or any(m["schema_rows_valid"] != m["of"] or m["plain_rows_nonempty"] != m["of"] for m in mixed.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
