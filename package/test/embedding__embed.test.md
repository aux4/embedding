# embedding embed

These tests download and run the real EmbeddingGemma 2 model (`onnx-community/embeddinggemma-2-ONNX`) on
CPU. The package installs its runtime dependencies (`@huggingface/transformers`, pulled on first use - see
the README) in addition to downloading the model, so the very first test in a clean environment pays for
both an `npm install` and a cold model download. The timeout below is generous to cover that worst case.

## text-only embedding

### should return one 768-d vector per input item

```timeout
300000
```

```execute
echo '["the cat sat on the mat", "a dog barked at the mailman"]' | aux4 embedding embed --modalities text --dtype q8 --cacheDir /tmp/aux4-embedding-test-cache
```

```expect:regex
^\[\[-?\d+\.?\d*(e-?\d+)?(,-?\d+\.?\d*(e-?\d+)?){767}\],\[-?\d+\.?\d*(e-?\d+)?(,-?\d+\.?\d*(e-?\d+)?){767}\]\]$
```

## dimension truncation

### should return a 128-d unit vector when --dimension 128 is given

```timeout
60000
```

```execute
echo '["short text"]' | aux4 embedding embed --modalities text --dtype q8 --dimension 128 --cacheDir /tmp/aux4-embedding-test-cache
```

```expect:regex
^\[\[-?\d+\.?\d*(e-?\d+)?(,-?\d+\.?\d*(e-?\d+)?){127}\]\]$
```

## relevance

### query should be more similar to a relevant document than an irrelevant one

```timeout
60000
```

```execute
echo '["task: search result | query: what is the capital of france?", "title: none | text: paris is the capital of france.", "title: none | text: bananas are a good source of potassium."]' | aux4 embedding embed --modalities text --task document --dtype q8 --cacheDir /tmp/aux4-embedding-test-cache
```

```expect:regex
^\[\[.*\],\[.*\],\[.*\]\]$
```

## invalid dimension

### should error on an unsupported --dimension value

```execute
echo '["hello"]' | aux4 embedding embed --dimension 999 --cacheDir /tmp/aux4-embedding-test-cache
```

```error:partial
Error: --dimension must be one of *?
```

## media requires full modality

### should error when an image is passed with --modalities text

```execute
echo '[{"image":"/tmp/does-not-matter.jpg"}]' | aux4 embedding embed --modalities text --cacheDir /tmp/aux4-embedding-test-cache
```

```error:partial
Error: item has media *?
```
