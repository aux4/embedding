# embedding info

Reads the cache directory on disk; does not load the model or touch the network. Depends on the
`embedding download` test having already populated `/tmp/aux4-embedding-test-cache`.

## after downloading the text-only model

### should report downloaded true with the requested dtype and dimension

```execute
aux4 embedding info --modalities text --dtype q8 --dimension 256 --cacheDir /tmp/aux4-embedding-test-cache
```

```expect:regex
^\{"model":"onnx-community/embeddinggemma-2-ONNX","modalities":"text","dtype":"q8","dimension":256,"nativeDimension":768,"cacheDir":".*","downloaded":true,"sizeBytes":\d+\}$
```

## before downloading anything

### should report downloaded false for an empty cache directory

```execute
aux4 embedding info --cacheDir /tmp/aux4-embedding-test-cache-empty
```

```expect:regex
^\{.*"downloaded":false,"sizeBytes":0\}$
```
