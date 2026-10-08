#### Description

Reads a JSON array from stdin and writes a JSON array of embedding vectors to stdout, using Google's
EmbeddingGemma 2 model running locally on CPU. Each stdin item is either:

- a plain string — embedded as text, or
- an object `{ "text": "...", "title": "...", "image": "/path/to/file.jpg" }` — any combination of `text`
  and `image` (an item-level `title` overrides the `--title` flag for the `document` task prefix).

All progress and status output is written to stderr, so stdout only ever contains the JSON result — safe to
pipe directly into another tool.

The model is loaded once per invocation and reused for every item in the batch. Images require
`--modalities full`; passing an `image`/`audio`/`video` field with `--modalities text` is an error.
`audio`/`video` fields are not implemented in this version (see the package README's Limitations section)
and always return an error.

The embedding text is prefixed per the EmbeddingGemma 2 prompt format before tokenization:

- `task: query` → `task: search result | query: <text>`
- `task: document` → `title: <title|none> | text: <text>`
- any other `task` value → best-effort `task: <task> | query: <text>` (not verified against the model card)

When `--dimension` is less than the model's native 768, the output vector is truncated to that length and
re-normalized (Matryoshka representation) — this preserves cosine-similarity ranking while shrinking storage.

#### Usage

```bash
echo '[...]' | aux4 embedding embed [--modalities text|full] [--task query|document|<other>] [--title <title>] [--dimension 768|512|256|128] [--dtype fp32|fp16|q8|q4|q4f16] [--cacheDir <dir>] [--device cpu]
```

--modalities  `text` (text-only model) or `full` (text+image). Default: `text`
--task        `query`, `document`, or any other task name. Default: `document`
--title       Default document title when an item doesn't carry its own `title`. Default: `none`
--dimension   `768`, `512`, `256`, or `128`. Default: `768`
--dtype       `fp32`, `fp16`, `q8`, `q4`, or `q4f16`. Default: `q8`
--cacheDir    Model cache directory. Default: `~/.aux4.config/cache/embedding`
--device      Inference device. Default: `cpu`

#### Example

```bash
echo '["the cat sat on the mat", "a dog barked at the mailman"]' | aux4 embedding embed --dtype q8
```

```text
[[0.0123,-0.045,...],[0.031,0.002,...]]
```

Embedding an image (requires `--modalities full`):

```bash
echo '[{"image":"/path/to/photo.jpg"}]' | aux4 embedding embed --modalities full
```
