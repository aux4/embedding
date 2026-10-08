#### Description

Reports the model id, the requested `--dtype`/`--dimension`, the cache directory, whether the model has
already been downloaded into that directory, and the on-disk size (in bytes) of the cached model files.
Does not load the model or run any inference — this is a cheap, offline status check.

#### Usage

```bash
aux4 embedding info [--modalities text|full] [--dimension 768|512|256|128] [--dtype fp32|fp16|q8|q4|q4f16] [--cacheDir <dir>]
```

--modalities  `text` or `full`. Default: `text`
--dimension   `768`, `512`, `256`, or `128`. Default: `768`
--dtype       `fp32`, `fp16`, `q8`, `q4`, or `q4f16`. Default: `q8`
--cacheDir    Model cache directory. Default: `~/.aux4.config/cache/embedding`

#### Example

```bash
aux4 embedding info
```

```text
{"model":"onnx-community/embeddinggemma-2-ONNX","modalities":"text","dtype":"q8","dimension":768,"nativeDimension":768,"cacheDir":"/Users/you/.aux4.config/cache/embedding","downloaded":true,"sizeBytes":346397233}
```
