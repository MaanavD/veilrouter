# veilroute

`veilroute` routes simple prompts to a local model and harder prompts to an
OpenAI-compatible cloud model. Cloud-bound requests are redacted before they
leave the machine and restored transparently in the model response.

```python
from veilroute import Router, RouterConfig

router = Router(RouterConfig(
    local_model="qwen2.5-0.5b",
    scorer_model="qwen2.5-0.5b",
    cloud_endpoint="https://example.openai.azure.com/openai/v1",
    cloud_api_key="...",
    cloud_model="gpt-4o",
    pii_model_path="path/to/LFM2.5-350M-Classifier-PII-Demo",
))

response = router.run("Summarize this contract for John Smith at john@example.com")
print(response.text)
print(response.route, response.score, response.cost_saved)
```

`run()` and `stream()` accept either a string or OpenAI-style chat messages.
The cloud API key is resolved from explicit config first, then
`VEILROUTE_CLOUD_API_KEY`. Secrets and raw PII are not included in telemetry or
debug logs.

## Provider notes

Live local execution requires Foundry Local and the optional SDK dependency:

```powershell
pip install -e ".[foundry]"
```

The cloud provider uses the OpenAI Python SDK with a configurable `base_url`, so
it works with OpenAI-compatible endpoints including Azure OpenAI, vLLM, and
local gateways.

If you are running from this workspace and keeping the demo PII model assets
outside the git repo, pass `pii_model_path=r"..\LFM2.5-350M-Classifier-PII-Demo"`.

## Offline benchmark harness

Run the lightweight core benchmark from the repository root:

```powershell
python benchmarks\benchmark_core.py --iterations 100
python benchmarks\benchmark_core.py --mode stream --iterations 25 --format json
```

The harness uses deterministic fake local/cloud providers, a heuristic scorer,
and the regex PII detector. It does not require live provider credentials,
Foundry Local, model downloads, or network access.
