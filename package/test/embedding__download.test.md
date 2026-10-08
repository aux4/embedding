# embedding download

Requires network access on first run to populate the cache.

## text-only download

### should report the model as downloaded

```timeout
180000
```

```execute
aux4 embedding download --modalities text --dtype q8 --cacheDir /tmp/aux4-embedding-test-cache
```

```expect:regex
^\{"model":"onnx-community/embeddinggemma-2-ONNX","modalities":"text","dtype":"q8","cacheDir":".*","downloaded":true\}$
```
