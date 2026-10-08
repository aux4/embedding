#!/usr/bin/env node
import fs from "fs";
import path from "path";

const MODEL_ID = "onnx-community/embeddinggemma-2-ONNX";
const NATIVE_DIM = 768;
const VALID_DIMENSIONS = [768, 512, 256, 128];

function log(...args) {
  console.error(...args);
}

function fail(message) {
  log(`Error: ${message}`);
  process.exit(1);
}

async function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", chunk => (data += chunk));
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

function resolveCacheDir(cacheDir) {
  const dir = path.resolve(cacheDir);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

async function getTransformers(cacheDir) {
  const transformers = await import("@huggingface/transformers");
  transformers.env.cacheDir = resolveCacheDir(cacheDir) + path.sep;
  transformers.env.allowLocalModels = false;
  return transformers;
}

async function loadModel({ modalities, dtype, cacheDir, device, progress }) {
  const transformers = await getTransformers(cacheDir);
  const { AutoConfig, AutoModel, AutoTokenizer, AutoProcessor } = transformers;

  const progress_callback = progress
    ? data => {
        if (data.status === "progress") {
          const pct = data.total ? Math.round((data.loaded / data.total) * 100) : 0;
          log(`downloading ${data.file}: ${pct}%`);
        } else if (data.status) {
          log(`${data.status}${data.file ? " " + data.file : ""}`);
        }
      }
    : undefined;

  const config = await AutoConfig.from_pretrained(MODEL_ID, { progress_callback });

  if (modalities === "text") {
    config.vision_config = null;
    config.audio_config = null;
  }
  // modalities === "full" keeps vision_config; audio_config left in place too,
  // since video/audio support in this package is unverified (see README).

  const model = await AutoModel.from_pretrained(MODEL_ID, {
    config,
    dtype,
    device,
    progress_callback
  });
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });

  let processor = null;
  if (modalities === "full") {
    processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback });
  }

  return { transformers, model, tokenizer, processor };
}

function buildPrefixedText(task, title, text) {
  if (text === undefined || text === null) {
    return null;
  }
  if (task === "query") {
    return `task: search result | query: ${text}`;
  }
  if (task === "document") {
    return `title: ${title || "none"} | text: ${text}`;
  }
  // Best-effort for any other README task name: "task: <name> | query: <text>".
  // Only "query" and "document" are documented/verified by the EMB-001 spike.
  return `task: ${task} | query: ${text}`;
}

function truncateAndNormalize(vector, dimension) {
  let v = vector;
  if (dimension < NATIVE_DIM) {
    v = vector.slice(0, dimension);
  }
  let norm = 0;
  for (const value of v) norm += value * value;
  norm = Math.sqrt(norm);
  if (norm === 0) return v;
  return v.map(value => value / norm);
}

async function embedItem({ item, task, title, dimension, transformers, model, tokenizer, processor, modalities }) {
  const isObject = typeof item === "object" && item !== null;
  const text = isObject ? item.text : item;
  const imagePath = isObject ? item.image : undefined;
  const audioPath = isObject ? item.audio : undefined;
  const videoPath = isObject ? item.video : undefined;
  const itemTitle = isObject && item.title !== undefined ? item.title : title;

  const prefixedText = buildPrefixedText(task, itemTitle, text);

  let output;
  if (imagePath || audioPath || videoPath) {
    if (modalities !== "full") {
      fail(`item has media (image/audio/video) but --modalities is "${modalities}"; use --modalities full`);
    }
    if (audioPath || videoPath) {
      fail("audio/video embedding is not implemented in this package yet (unverified in EMB-001 spike); only image media is supported for --modalities full");
    }
    const { RawImage } = transformers;
    const image = await RawImage.read(imagePath);
    const inputs = await processor(prefixedText ?? null, [[image]]);
    output = await model(inputs);
  } else {
    const inputs = await tokenizer([prefixedText], { padding: true });
    output = await model(inputs);
  }

  const vector = Array.from(output.sentence_embedding.tolist()[0]);
  return truncateAndNormalize(vector, dimension);
}

async function cmdEmbed({ modalities, task, title, dimension, dtype, cacheDir, device }) {
  if (!VALID_DIMENSIONS.includes(dimension)) {
    fail(`--dimension must be one of ${VALID_DIMENSIONS.join(", ")}`);
  }

  const raw = await readStdin();
  let items;
  try {
    items = JSON.parse(raw);
  } catch (e) {
    fail(`stdin is not valid JSON: ${e.message}`);
  }
  if (!Array.isArray(items)) {
    fail("stdin must be a JSON array of strings or {text,image,audio,video} objects");
  }

  log(`loading ${MODEL_ID} (modalities=${modalities}, dtype=${dtype}, device=${device})...`);
  const t0 = Date.now();
  const { transformers, model, tokenizer, processor } = await loadModel({ modalities, dtype, cacheDir, device, progress: true });
  log(`model loaded in ${Date.now() - t0}ms`);

  const results = [];
  for (const item of items) {
    const vector = await embedItem({ item, task, title, dimension, transformers, model, tokenizer, processor, modalities });
    results.push(vector);
  }

  process.stdout.write(JSON.stringify(results));
}

async function cmdDownload({ modalities, dtype, cacheDir, device }) {
  log(`downloading ${MODEL_ID} (modalities=${modalities}, dtype=${dtype}) into ${path.resolve(cacheDir)}...`);
  const t0 = Date.now();
  await loadModel({ modalities, dtype, cacheDir, device, progress: true });
  log(`done in ${Date.now() - t0}ms`);
  process.stdout.write(
    JSON.stringify({
      model: MODEL_ID,
      modalities,
      dtype,
      cacheDir: path.resolve(cacheDir),
      downloaded: true
    })
  );
}

function dirSizeBytes(dir) {
  let total = 0;
  if (!fs.existsSync(dir)) return 0;
  const stack = [dir];
  while (stack.length) {
    const current = stack.pop();
    const stat = fs.statSync(current);
    if (stat.isDirectory()) {
      for (const entry of fs.readdirSync(current)) {
        stack.push(path.join(current, entry));
      }
    } else {
      total += stat.size;
    }
  }
  return total;
}

async function cmdInfo({ modalities, dimension, dtype, cacheDir }) {
  const resolvedCacheDir = path.resolve(cacheDir);
  const modelDir = path.join(resolvedCacheDir, MODEL_ID);
  const downloaded = fs.existsSync(path.join(modelDir, "onnx", `model_quantized.onnx`)) || fs.existsSync(path.join(modelDir, "onnx"));
  const sizeBytes = dirSizeBytes(modelDir);

  process.stdout.write(
    JSON.stringify({
      model: MODEL_ID,
      modalities,
      dtype,
      dimension,
      nativeDimension: NATIVE_DIM,
      cacheDir: resolvedCacheDir,
      downloaded,
      sizeBytes
    })
  );
}

async function main() {
  const [, , subcommand, ...args] = process.argv;

  if (subcommand === "embed") {
    const [modalities, task, title, dimensionStr, dtype, cacheDir, device] = args;
    await cmdEmbed({
      modalities,
      task,
      title,
      dimension: parseInt(dimensionStr, 10),
      dtype,
      cacheDir,
      device
    });
  } else if (subcommand === "download") {
    const [modalities, dtype, cacheDir, device] = args;
    await cmdDownload({ modalities, dtype, cacheDir, device });
  } else if (subcommand === "info") {
    const [modalities, dimensionStr, dtype, cacheDir] = args;
    await cmdInfo({ modalities, dimension: parseInt(dimensionStr, 10), dtype, cacheDir });
  } else {
    fail(`unknown subcommand: ${subcommand}`);
  }
}

main().catch(e => {
  log("Error:", e.message);
  log(e.stack);
  process.exit(1);
});
