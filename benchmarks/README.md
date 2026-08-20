# Benchmarks — moved

Inference benchmarking for this stack now lives in its own repository:

**→ [wu840407/llm-inference-benchmark](https://github.com/wu840407/llm-inference-benchmark)**

vLLM vs TensorRT-LLM on a single RTX 3090, with the measured noise floor reported
alongside the engine comparison and every figure cross-checked against a
first-principles roofline.

Headline results:
- Engine-to-engine difference (~2%) sits below the run-to-run noise floor (5.4%)
- TensorRT-LLM's ahead-of-time backend costs 1225 s to start and is 18.5% slower at batch 32
- Both engines run at 83–98% of the hardware roofline — bandwidth bounds decode, compute bounds prefill
