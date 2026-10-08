# aux4/embedding

Local, offline CPU embeddings using Google's EmbeddingGemma 2 model. Runs entirely on your machine — no API
key, no network calls once the model is downloaded — and produces 768-dimensional vectors (with optional
Matryoshka truncation to 512/256/128) that can be used for RAG, semantic search, or clustering. Published on
[hub.aux4.io](https://hub.aux4.io/aux4/embedding).

## Installation

```bash
aux4 aux4 pkger install aux4/embedding
```

The first `embed` or `download` call fetches the model (a few hundred MB depending on `--dtype` and
`--modalities`) and caches it for later runs.

## Quick Start

```bash
echo '["the cat sat on the mat", "a dog barked at the mailman"]' | aux4 embedding embed
```

Output is a JSON array of float vectors, one per input item, in the same order:

```json
[[0.0123, -0.045, ...], [0.031, 0.002, ...]]
```

## Commands

### embedding embed

Reads a JSON array from stdin — each item is either a plain string or an object
`{ "text": "...", "image": "path/to/file.jpg" }` — and writes a JSON array of embedding vectors to stdout.
All progress/log output goes to stderr, so stdout is always clean, parseable JSON.

```bash
echo '["hello world"]' | aux4 embedding embed --modalities text --task document --dimension 768 --dtype q8
```

```yaml
config:
  embeddings:
    modalities: text    # default: text
    task: document       # default: document
    dimension: 768        # default: 768
    dtype: q8              # default: q8
    cacheDir: ~/.aux4.config/cache/embedding
    device: cpu            # default: cpu
```

| Variable | Default | Description |
|---|---|---|
| `modalities` | `text` | `text` (text-only model, smaller/faster) or `full` (text + image) |
| `task` | `document` | `query` or `document` — selects the EmbeddingGemma prompt prefix. Other task names are passed through best-effort as `task: <name> \| query: <text>` |
| `title` | `none` | Document title used for the `document` task prefix, when an item doesn't carry its own `title` |
| `dimension` | `768` | `768`, `512`, `256`, or `128` — truncates the native 768-d vector and re-normalizes (Matryoshka) |
| `dtype` | `q8` | Model quantization: `fp32`, `fp16`, `q8`, `q4`, `q4f16` |
| `cacheDir` | `~/.aux4.config/cache/embedding` | Where the model is downloaded/cached |
| `device` | `cpu` | Inference device |

**Input item shapes:**

- A plain string → embedded as text with the `task`/`title` prefix applied.
- `{ "text": "..." }` → same as a string, with an optional per-item `"title"` overriding the default.
- `{ "image": "/path/to/file.jpg" }` → embedded as an image. Requires `--modalities full`.
- `{ "text": "...", "image": "/path/to/file.jpg" }` → text and image embedded jointly. Requires `--modalities full`.

**Note:** text and image embeddings share the same vector space (same model), so they are directly
comparable with cosine similarity — this is what makes EmbeddingGemma 2 useful for "find the image that
matches this text" style search, in addition to plain text RAG.

**Audio and video are not implemented yet.** The upstream model supports them, but this package has only
been verified for text and image — passing `"audio"` or `"video"` fields currently returns an error. See
Limitations below.

### embedding download

Pre-fetches the model into the cache directory without embedding anything — useful to warm the cache ahead
of time (e.g. during a build/deploy step) instead of paying the download cost on the first real
`embedding embed` call.

```bash
aux4 embedding download --modalities text --dtype q8
```

| Variable | Default | Description |
|---|---|---|
| `modalities` | `text` | `text` or `full` |
| `dtype` | `q8` | Model quantization to download |
| `cacheDir` | `~/.aux4.config/cache/embedding` | Where to download the model |
| `device` | `cpu` | Inference device |

### embedding info

Reports the model id, configured dtype/dimension, cache directory, whether the model has already been
downloaded, and the on-disk size of the cached model files.

```bash
aux4 embedding info
```

```json
{
  "model": "onnx-community/embeddinggemma-2-ONNX",
  "modalities": "text",
  "dtype": "q8",
  "dimension": 768,
  "nativeDimension": 768,
  "cacheDir": "/Users/you/.aux4.config/cache/embedding",
  "downloaded": true,
  "sizeBytes": 346397233
}
```

## Choosing a dtype

The model ships in several quantizations (text-only component size, cosine similarity vs the full-precision
`fp32` baseline measured on text):

| dtype | size | cosine vs fp32 |
|---|---|---|
| `fp32` | ~1085 MB | 1.0 (baseline) |
| `fp16` | ~543 MB | ~1.0 |
| `q8` (default) | ~314 MB | ~0.9997 |
| `q4` | ~175 MB | ~0.988 |
| `q4f16` | ~157 MB | ~0.988 |

`q8` is the default: effectively lossless similarity at under a third of the `fp32` size. Use `q4`/`q4f16`
only if disk/memory is tight and a small similarity hit is acceptable.

## Using with aux4/ai-agent RAG

`aux4/ai-agent`'s `learn`/`search` commands default to OpenAI embeddings. To use `aux4/embedding` instead for
a fully local, offline pipeline, run your documents through `aux4 embedding embed` and feed the resulting
vectors into your own vector store, or pipe its output alongside your RAG indexing step. There is no
built-in `ai-agent` provider for `aux4/embedding` yet — that integration is a candidate follow-up, not
shipped here.

## Limitations

- **`--modalities full` (image embedding) only works on the platform the package was built/published for.**
  `sharp`, the native image-decoding dependency, installs only the binary for the build host's
  platform/architecture (an npm `optionalDependencies` limitation). The published build runs on a Linux CI
  runner, so **image embedding currently only works on Linux x64**; `text`-only embedding has no such
  restriction and works on every platform (it never loads `sharp`).
- **Audio and video are not implemented.** Only `text` and `image` item fields are supported in this version.
- **Items are embedded one at a time internally** (the model is loaded once per `embedding embed` invocation,
  but each item gets its own forward pass rather than a single padded batch). This keeps mixed
  text/image/absent inputs simple and correct; very large arrays will be proportionally slower than a true
  batched call.
- **`task` values other than `query`/`document`** are passed through as a best-effort `task: <name> | query:
  <text>` prefix. Only `query` and `document` prefixes have been verified against the model card.

## Environment

The model is downloaded from the Hugging Face Hub (`onnx-community/embeddinggemma-2-ONNX`) on first use. No
API key is required — the model is Apache-2.0 licensed and ungated.
