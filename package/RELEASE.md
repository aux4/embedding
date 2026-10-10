# Release notes

## 0.1.3

### Tiny package, runtime deps installed on your machine (EMB-006)

`package/lib/node_modules` is no longer shipped in the published package (it was ~263MB, making the zip
~95MB). The runtime dependencies (`@huggingface/transformers`, including the native ONNX Runtime and, for
`--modalities full`, `sharp`) are now installed on your machine on first use of `embed` or `download` —
this gets you the correct native binaries for your own platform/architecture, so `--modalities full` (image
embedding) now works everywhere, not just the platform the package happened to be built on. `npm install`
is added to the `system` requirements check alongside `node`. Installation is concurrency-safe (guarded by
a lock file) and all progress goes to stderr, never stdout. `onnxruntime-web` (the unused browser backend)
is pruned after install. `package/lib/package-lock.json` is now committed for reproducible installs.

## 0.1.0

### Local CPU embeddings with EmbeddingGemma 2 (EMB-001, EMB-002)

First release. Runs Google's EmbeddingGemma 2 (`onnx-community/embeddinggemma-2-ONNX`, Apache-2.0,
ungated) locally on CPU via `@huggingface/transformers`/ONNX Runtime — no API key, no network calls after
the model is cached.

- **`aux4 embedding embed`** — stdin a JSON array of strings or `{text,title,image}` objects, stdout a JSON
  array of embedding vectors. `--modalities text|full` selects text-only or text+image. `--dimension
  768|512|256|128` truncates and re-normalizes (Matryoshka). `--dtype` defaults to `q8` (near-lossless vs
  `fp32` at ~1/3 the size). `--task query|document` applies the EmbeddingGemma prompt prefix.
- **`aux4 embedding download`** — pre-fetches the model for a given modalities/dtype combination.
- **`aux4 embedding info`** — reports model id, dtype, dimension, cache directory, and download status
  without touching the network.
- Audio and video item fields are recognized but not implemented yet; they return an explicit error rather
  than silently embedding nothing.
