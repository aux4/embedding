#### Description

The `embedding` command group provides local, offline CPU embeddings using Google's EmbeddingGemma 2 model
(`onnx-community/embeddinggemma-2-ONNX`, Apache-2.0, ungated). All inference runs on your machine via ONNX
Runtime — no API key, no network calls once the model is cached.

The published package does not ship its runtime dependencies. The first `embed` or `download` call installs
them (`@huggingface/transformers` and its native ONNX Runtime/`sharp` binaries, matched to your
platform/architecture) and then downloads the model — both need `npm` and network access, and happen once.

Subcommands:

- **`embed`** — embed a batch of text/image items (stdin in, JSON vectors out)
- **`download`** — pre-fetch the model without embedding anything
- **`info`** — report model/cache status

#### Usage

```bash
aux4 embedding <embed|download|info> [options]
```

#### Example

```bash
echo '["hello world"]' | aux4 embedding embed
```

```text
[[0.0123,-0.045, ...]]
```
