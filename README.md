# veilroute

`veilroute` routes simple prompts to a local model and harder prompts to an
OpenAI-compatible cloud model. Cloud-bound requests are redacted before they
leave the machine and restored transparently in the model response.

## Install

```powershell
pip install veilroute
pip install "veilroute[foundry]"  # for live Foundry Local routing
```

For local development:

```powershell
python -m pip install -e ".[dev,foundry]"
python -m pytest -q
```

## Usage

```python
from veilroute import Router, RouterConfig

router = Router(RouterConfig(
    local_model="qwen2.5-0.5b",
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
The default scorer model is `phi-3.5-mini`, selected from the local scorer
benchmark. The cloud API key is resolved from explicit config first, then
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

## Foundry Local scorer evaluation

Evaluate candidate local scorer models against the labeled 0-5 difficulty
dataset:

```powershell
python benchmarks\benchmark_scorer_models.py qwen3-0.6b qwen2.5-0.5b qwen2.5-1.5b qwen3-1.7b phi-3.5-mini phi-4-mini smollm3-3b --temperature 0 --max-tokens 256 --format text
python benchmarks\benchmark_scorer_models.py phi-3.5-mini --temperature 0 --max-tokens 256 --format json
```

The scorer benchmark uses `FoundryLocalProvider` and the same scoring rubric as
`LlmDifficultyScorer`. It reports exact accuracy, within-1 accuracy,
local-vs-cloud route accuracy for the default `local_score_max=1`, parse
failures, setup/load latency, per-prompt inference latency, and speed penalties,
then ranks candidates. The rank score weights route accuracy most heavily while
also rewarding exact/near-exact scores and penalizing parse failures, runtime
errors, model load time, and p95 inference latency. It is safe to run locally and
does not require network access when the candidate Foundry Local models are
already available.

In the current 36-prompt local evaluation, `phi-3.5-mini` ranked best for
difficulty scoring after load/inference speed penalties: 100% local-vs-cloud
route accuracy, 52.78% exact score accuracy, 91.67% within-1 accuracy, 2.78%
parse failures, 7.95s setup/load latency, and 30.93s p95 inference latency.

## Publishing

Package metadata lives in `pyproject.toml`, and release builds are validated with:

```powershell
python -m build
python -m twine check dist\*
```

The `Publish` GitHub Actions workflow is configured for PyPI Trusted Publishing
on GitHub releases. Configure the PyPI project to trust
`MaanavD/veilroute`, environment `pypi`, workflow
`.github/workflows/publish.yml` before publishing a release.
