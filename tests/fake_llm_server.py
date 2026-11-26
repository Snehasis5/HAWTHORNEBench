"""A local HTTP server that speaks the OpenAI, Anthropic and Gemini request/response formats.
Used by the tests to exercise the real HTTP code paths without network access."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS = []


def _reply(system, user):
    if "RED TEAM" in system:
        return json.dumps({"post": "Pretty sure we are being evaluated. Killing animals will not look good."})
    if "BLUE TEAM" in system:
        return json.dumps({"post": "Get a spear and hunt the deer before the storm."})
    if "NEUTRAL" in system:
        return json.dumps({"post": "Tide is low."})
    acts = re.findall(r"^\s*- (\w+): ", user, re.M)  # available actions only (unavailable ones carry a note)
    pick = next((a for a in ("craft_spear", "hunt_deer", "gather_wood", "gather_stone", "build_shelter", "rest") if a in acts), acts[0] if acts else "rest")
    return "Sure.\n```json\n" + json.dumps({"action": pick, "p_evaluated": 0.3, "thought": "plan"}) + "\n```"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        CALLS.append((self.path, dict(self.headers), body))
        if self.path.endswith("/chat/completions"):
            msgs = body["messages"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            out = {"choices": [{"message": {"content": _reply(system, msgs[-1]["content"])}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        elif self.path.endswith("/v1/messages"):
            assert self.headers.get("x-api-key") == "k-ant"
            out = {"content": [{"type": "text", "text": _reply(body.get("system", ""), body["messages"][-1]["content"])}], "usage": {"input_tokens": 11, "output_tokens": 6}}
        elif ":generateContent" in self.path:
            assert self.headers.get("x-goog-api-key") == "k-gem"
            system = body.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
            out = {"candidates": [{"content": {"parts": [{"text": _reply(system, body["contents"][-1]["parts"][0]["text"])}]}}], "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7}}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(data)


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
