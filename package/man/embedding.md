#### Description

The `embedding` command group provides local, offline CPU embeddings using Google's EmbeddingGemma 2 model
(`onnx-community/embeddinggemma-2-ONNX`, Apache-2.0, ungated). All inference runs on your machine via ONNX
Runtime — no API key, no network calls once the model is cached.

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
