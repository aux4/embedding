#### Description

Pre-fetches the EmbeddingGemma 2 model into the cache directory for the given `--modalities`/`--dtype`
combination, without embedding anything. Useful for warming the cache ahead of time — e.g. in a build or
deploy step — so the first real `aux4 embedding embed` call doesn't pay the download cost.

Downloading `--modalities full` additionally fetches the vision encoder component on top of the text
component already required by `--modalities text`.

#### Usage

```bash
aux4 embedding download [--modalities text|full] [--dtype fp32|fp16|q8|q4|q4f16] [--cacheDir <dir>] [--device cpu]
```

--modalities  `text` or `full`. Default: `text`
--dtype       `fp32`, `fp16`, `q8`, `q4`, or `q4f16`. Default: `q8`
--cacheDir    Model cache directory. Default: `~/.aux4.config/cache/embedding`
--device      Inference device. Default: `cpu`

#### Example

```bash
aux4 embedding download --modalities text --dtype q8
```

```text
{"model":"onnx-community/embeddinggemma-2-ONNX","modalities":"text","dtype":"q8","cacheDir":"/Users/you/.aux4.config/cache/embedding","downloaded":true}
```
