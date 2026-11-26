# Contributing

```bash
make install     # editable install with test extras
make test        # everything runs offline: mock and scripted agents, a fake HTTP provider
```

- **New provider:** a `ChatModel` subclass registered in `PROVIDERS` (see [guide/PROVIDERS.md](guide/PROVIDERS.md)), plus a test against `tests/fake_llm_server.py`.
- **New world mechanic or action:** update `world.py`, the trace schema if fields change, and the action table in the README.
- **New measure:** add it to `metrics.py` and to `experiment.PAIR_MEASURES`, so it gets the full statistics automatically; describe it in [guide/METHODOLOGY.md](guide/METHODOLOGY.md).
- **New prompt style:** add it to `prompts.py`, then regenerate `guide/PROMPTS.md`.
- **Traces:** if you change the trace format, bump the schema id (`hawthornebench/trace@N`) and update `schemas/trace.schema.json`. The tests validate every trace they produce against it.

Never commit API keys. Configs refer to keys by environment variable only (`api_key_env`).
