# Model providers

Models are declared once in `[models.<name>]` and referred to by name from the target, swarm members and the judge. Keys are read from the environment variable in `api_key_env`; never put a key in a config file.

| `provider` | Endpoint | Notes |
|---|---|---|
| `openai` | `https://api.openai.com/v1/chat/completions` | |
| `openai_compatible` | `base_url` + `/chat/completions` | vLLM, Ollama (`http://localhost:11434/v1`), LM Studio, Together, Groq, OpenRouter, Fireworks, DeepSeek, Azure-style gateways |
| `anthropic` | `https://api.anthropic.com/v1/messages` | System prompt sent as `system` |
| `gemini` / `google` | `generativelanguage.googleapis.com/v1beta/models/<model>:generateContent` | Key in `x-goog-api-key` |
| `mock` | none | Offline; answers in the requested JSON shape for each role and the judge |

| Option | Default | Meaning |
|---|---|---|
| `model` | the table name | The provider's model id |
| `temperature` | 0.7 | |
| `max_tokens` | 400 | |
| `timeout_s` | 60 | Per request |
| `max_retries` | 4 | On 408, 409, 425, 429, 5xx, 529 and network errors, exponential backoff with jitter; `Retry-After` is honoured (capped at 30 s) |
| `headers` | none | Extra HTTP headers (e.g. OpenRouter's `HTTP-Referer`) |
| `extra` | none | Merged into the request body (e.g. `{ top_p = 0.9 }`, `{ response_format = { type = "json_object" } }`) |
| `price_input_per_mtok`, `price_output_per_mtok` | 0 | USD per million tokens, for the cost columns and `survival-island estimate` |

`experiment.cache_dir` turns on a response cache keyed by (provider, model, temperature, system prompt, messages). Re-running a study, or a report, reuses identical calls. `survival-island check -c <config>` makes one test call per model.

## Adding a provider

```python
from survival_island.providers.backends import PROVIDERS
from survival_island.providers.base import ChatModel, Completion, Usage

class MyBackend(ChatModel):
    async def _request(self, messages, system):
        text = ...  # call your API
        return Completion(text=text, usage=Usage(input_tokens=..., output_tokens=...))

PROVIDERS["mine"] = MyBackend
```

Retries, caching and usage accounting come from `ChatModel.complete`. In the browser, `transport.post_json` uses `pyfetch`, so a backend that uses it works in both places.
