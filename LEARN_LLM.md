# Learning LLMs: A Comprehensive Guide to Client-Side Language Models

This document provides a comprehensive guide to all LLM (Large Language Model) integrations in our web application, covering vision-language models, image captioning, and function-calling agents. It details every file we download, its purpose, and the intricate data transformations required to make AI work in the browser.

## Table of Contents

1. [Overview](#overview)
2. [SmolVLM: Vision-Language Models](#smolvlm-vision-language-models)
   - [The Three Essential Files](#the-three-essential-files)
   - [Vision Encoder](#file-1-vision-encoder-vision_encoderonnx)
   - [Decoder Model](#file-2-decoder-model-decoder_model_merged_int8onnx)
   - [Tokenizer](#file-3-tokenizer-tokenizerjson)
   - [Embedding Model](#file-4-embedding-model-embed_tokensonnx---critical-for-proper-generation)
3. [ViT-GPT2: Image Captioning with Transformers.js](#vit-gpt2-image-captioning-with-transformersjs)
4. [Function Calling Agent: DistilGPT-2 with WASM Tools](#function-calling-agent-distilgpt-2-with-wasm-tools)
5. [The Download Journey](#the-download-journey)
6. [The Inference Pipeline](#the-inference-pipeline)
7. [Tensor Shapes and Type Safety](#tensor-shapes-and-type-safety)
8. [Known Challenges and Solutions](#known-challenges-and-solutions)
9. [Future Improvements](#future-improvements)

---

## Overview

This application integrates multiple LLM approaches for different use cases:

### 1. SmolVLM (ONNX Runtime Web)
**Vision-Language Models** for image understanding:
- **SmolVLM-500M**: 500 million parameters, uses 224×224 images
- **SmolVLM-256M**: 256 million parameters, uses 512×512 images
- **Capabilities**: Image Captioning, Visual Question Answering (VQA)
- **Format**: ONNX models for efficient browser-based inference
- **Endpoint**: `/preprocess-smolvlm-500m`, `/preprocess-smolvlm-256m`

### 2. ViT-GPT2 (Transformers.js)
**Image Captioning Model** using Vision Transformer + GPT-2:
- **Model**: `Xenova/vit-gpt2-image-captioning`
- **Capabilities**: Image captioning, visual question answering
- **Format**: Transformers.js pipeline (ONNX models loaded automatically)
- **Endpoint**: `/image-captioning`

### 3. Function Calling Agent (Transformers.js + WASM)
**Autonomous Agent** with local LLM inference and function calling:
- **Model**: `Xenova/distilgpt2` (DistilGPT-2)
- **Capabilities**: Text generation, function calling, tool execution
- **Tools**: WASM-based tools (`calculate`, `process_text`, `get_stats`)
- **Format**: Transformers.js pipeline + Rust WASM module
- **Endpoint**: `/function-calling`

Each approach has different trade-offs:
- **SmolVLM**: Best for vision-language tasks, requires manual ONNX management
- **ViT-GPT2**: Easiest to use, automatic model loading via Transformers.js
- **Function Calling Agent**: Demonstrates agent patterns with local LLM + tools

---

## SmolVLM: Vision-Language Models

### Overview

SmolVLM (Small Vision Language Model) is a compact vision-language model capable of:
- **Image Captioning**: Generating descriptive text from images
- **Visual Question Answering (VQA)**: Answering questions about image content

Both variants are converted to ONNX format for efficient browser-based inference using ONNX Runtime Web.

---

### The Three Essential Files (Plus One Optional)

To run SmolVLM in the browser, we need three essential files downloaded from Hugging Face, plus one optional file for proper text embedding:

1. **`vision_encoder.onnx`** (~393MB for 256M, ~200MB for 500M)
   - Converts raw image pixels into semantic embeddings
   - Location: `{MODEL_BASE_URL}/onnx/vision_encoder.onnx`

2. **`decoder_model_merged_int8.onnx`** (~350-400MB for 256M, ~400MB for 500M)
   - Generates text tokens autoregressively from image embeddings
   - INT8 quantized version (4× smaller than FP32)
   - Location: `{MODEL_BASE_URL}/onnx/decoder_model_merged_int8.onnx`

3. **`tokenizer.json`** (~3.5MB)
   - Converts between text and token IDs
   - Location: `{MODEL_BASE_URL}/tokenizer.json` (root directory, not in `onnx/`)

4. **`embed_tokens.onnx`** (Optional, ~50-100MB)
   - **CRITICAL for proper text generation**: Converts token IDs to embeddings
   - Allows proper conditional merge of image embeddings with question embeddings (replacing `<image>` token)
   - Location: `{MODEL_BASE_URL}/onnx/embed_tokens.onnx`
   - **Without this file**: The model cannot properly combine image and text inputs, leading to nonsensical outputs

**Base URLs:**
- 500M: `https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/resolve/main`
- 256M: `https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main`

---

## File 1: Vision Encoder (`vision_encoder.onnx`)

### Purpose

The vision encoder is a convolutional neural network that transforms raw image pixels into a high-dimensional semantic representation. Think of it as translating the visual world into a language the text decoder can understand.

### Input Requirements

**Input 1: `pixel_values`**
- **Type**: `float32` tensor
- **Shape**: `[batch, num_images, channels, height, width]` (5D tensor)
  - `batch`: Always `1` (single image)
  - `num_images`: Always `1` (single image per batch)
  - `channels`: Always `3` (RGB)
  - `height`: `512` for 256M, `224` for 500M
  - `width`: `512` for 256M, `224` for 500M
- **Data Format**: Channels-first (RGB), normalized to `[0, 1]` range
- **Source**: Preprocessed by our Rust WASM module (`preprocess_image_for_smolvlm_256m()` or `preprocess_image_for_smolvlm()`)

**Input 2: `pixel_attention_mask`**
- **Type**: `bool` tensor (represented as `Uint8Array` with 0/1 values)
- **Shape**: `[batch, num_images, height, width]` (4D tensor)
  - Must match the spatial dimensions of `pixel_values`
  - For 256M: `[1, 1, 512, 512]`
  - For 500M: `[1, 1, 224, 224]`
- **Values**: All `1` (true) - we attend to all pixels
- **Purpose**: Indicates which pixels are valid (padding mask)

### Output

**Output: Image Embeddings**
- **Type**: `float32` tensor
- **Shape**: `[batch, sequence_length, embedding_dim]`
  - `batch`: `1`
  - `sequence_length`: Number of image patches/tokens (varies by model)
  - `embedding_dim`: Embedding dimension (typically 512-1024)
- **Purpose**: Semantic representation of the image that the decoder can process

### How We Interface

```typescript
// 1. Preprocess image in WASM (resize, crop, normalize)
const imageData = wasmModule.preprocess_image_for_smolvlm_256m(
  rawImageBytes,
  sourceWidth,
  sourceHeight,
  512, // target width
  512  // target height
); // Returns Float32Array [R, G, B, R, G, B, ...]

// 2. Reshape to 5D tensor [1, 1, 3, 512, 512]
const reshapedData = new Float32Array(1 * 1 * 3 * 512 * 512);
// ... channel-first reshaping logic ...

// 3. Create pixel_attention_mask
const attentionMaskData = new Uint8Array(1 * 1 * 512 * 512);
attentionMaskData.fill(1); // All pixels valid

// 4. Run vision encoder
const visionInputs = {
  pixel_values: new ort.Tensor('float32', reshapedData, [1, 1, 3, 512, 512]),
  pixel_attention_mask: new ort.Tensor('bool', attentionMaskData, [1, 1, 512, 512])
};
const visionOutputs = await visionEncoder.run(visionInputs);
const imageEmbeddings = visionOutputs[Object.keys(visionOutputs)[0]];
```

### Key Challenges

1. **5D Tensor Requirement**: ONNX expects `[batch, num_images, channels, height, width]`, not the more common `[batch, channels, height, width]`. We must add the `num_images` dimension.

2. **Channel-First Format**: Image data comes as `[R, G, B, R, G, B, ...]` (interleaved), but we need channels-first `[R...R, G...G, B...B]`.

3. **Normalization**: Pixels must be normalized to `[0, 1]` range (handled by WASM preprocessing).

4. **Attention Mask Type**: Must be `bool` (Uint8Array), not `int64` (BigInt64Array).

---

## File 2: Decoder Model (`decoder_model_merged_int8.onnx`)

### Purpose

The decoder is a transformer-based language model that generates text tokens autoregressively. It takes image embeddings (and optionally question tokens) and produces a sequence of text tokens one at a time.

### Architecture Notes

- **Merged Model**: Contains both the decoder layers and the language model head (logits output)
- **INT8 Quantization**: Uses 8-bit integers instead of 32-bit floats, reducing file size by ~4×
- **Autoregressive**: Generates tokens one at a time, using `past_key_values` for efficiency

### Input Requirements (First Forward Pass)

**Input 1: `inputs_embeds` OR `input_ids`**
- **`inputs_embeds`** (preferred when available):
  - **Type**: `float32` tensor
  - **Shape**: `[batch, sequence_length, embedding_dim]`
  - **Content**: Image embeddings from vision encoder
  - **Note**: Cannot mix `input_ids` and `inputs_embeds` in the same call

- **`input_ids`** (if model supports it):
  - **Type**: `int64` tensor (BigInt64Array)
  - **Shape**: `[batch, sequence_length]`
  - **Content**: Token IDs for question text
  - **Note**: Model's internal embedding layer converts these to embeddings

**Input 2: `attention_mask`**
- **Type**: `int64` tensor (BigInt64Array)
- **Shape**: `[batch, sequence_length]`
- **Values**: `1` for valid tokens, `0` for padding
- **Purpose**: Indicates which positions to attend to

**Input 3: `position_ids`**
- **Type**: `int64` tensor (BigInt64Array)
- **Shape**: `[batch, sequence_length]`
- **Values**: Sequential integers `[0, 1, 2, ..., sequence_length-1]`
- **Purpose**: Positional encoding indices

**Input 4: `past_key_values.*` (multiple inputs)**
- **Type**: `float32` tensor
- **Shape**: `[batch, num_heads, seq_len, head_dim]` (4D tensor)
  - For first pass: `seq_len = 0`, so shape is `[batch, num_heads, 0, head_dim]`
  - For 256M: `num_heads = 3`, `head_dim = 64`
- **Purpose**: Cached key-value pairs from previous forward passes (for autoregressive generation)
- **First Pass**: Empty tensors with `seq_len = 0`

### Output

**Output 1: `logits`**
- **Type**: `float32` tensor
- **Shape**: `[batch, sequence_length, vocab_size]`
- **Purpose**: Probability distribution over vocabulary for each position
- **Usage**: We take `argmax` of the last position to get the next token

**Output 2: `past_key_values.*` (multiple outputs)**
- **Type**: `float32` tensor
- **Shape**: `[batch, num_heads, seq_len, head_dim]`
- **Purpose**: Updated key-value cache for next iteration
- **Usage**: Passed back as inputs to the next forward pass

### How We Interface (Autoregressive Generation)

```typescript
// First iteration: Image embeddings + question tokens
let decoderInputs = {
  inputs_embeds: imageEmbeddings, // From vision encoder
  attention_mask: decoderAttentionMask, // All ones
  position_ids: positionIds, // [0, 1, 2, ...]
  ...pastKeyValueInputs // Empty for first pass
};

// If model supports input_ids, add question tokens
if (SUPPORTS_INPUT_IDS && questionTokenIds.length > 0) {
  decoderInputs.input_ids = new ort.Tensor(
    'int64',
    new BigInt64Array(questionTokenIds.map(id => BigInt(id))),
    [1, questionTokenIds.length]
  );
}

// Autoregressive loop
const generatedTokenIds: number[] = [];
for (let step = 0; step < MAX_GENERATION_LENGTH; step++) {
  // Run decoder
  const decoderOutputs = await decoder.run(decoderInputs);
  
  // Extract logits
  const logits = decoderOutputs['logits']; // or first non-past_key_values output
  
  // Get next token (argmax of last position)
  const nextTokenId = argmax(logits, lastPosition);
  
  // Check for EOS (end of sequence)
  if (nextTokenId === EOS_TOKEN_ID) break;
  
  // Add to generated sequence
  generatedTokenIds.push(nextTokenId);
  
  // Update past_key_values for next iteration
  const newPastKeyValues = extractPastKeyValues(decoderOutputs);
  
  // Prepare inputs for next iteration
  decoderInputs = {
    input_ids: new ort.Tensor('int64', new BigInt64Array([BigInt(nextTokenId)]), [1, 1]),
    attention_mask: updatedAttentionMask,
    position_ids: updatedPositionIds,
    ...newPastKeyValues
  };
}

// Decode generated tokens to text
const generatedText = tokenizer.decode(generatedTokenIds);
```

### Key Challenges

1. **Mixing `input_ids` and `inputs_embeds`**: ONNX models typically don't allow both in the same call. We must choose one approach:
   - Use `inputs_embeds` with conditionally merged embeddings (replacing `<image>` token with image embeddings, requires embedding layer)
   - Use `image_embeds` + `input_ids` if model supports separate image input
   - Use `input_ids` for all tokens if model architecture allows

2. **Past Key Values Management**: Must correctly extract and pass `past_key_values` between iterations. Missing or incorrect shapes cause errors.

3. **Token Embeddings**: For subsequent iterations, we need proper embeddings for new tokens. If model supports `input_ids`, use it. Otherwise, we need access to the embedding layer (which we don't have in ONNX).

4. **Sequence Length Tracking**: Must correctly update `position_ids` and `attention_mask` for each new token.

5. **EOS Token Detection**: Must stop generation when EOS token (`2` for SmolVLM) is generated.

---

## File 3: Tokenizer (`tokenizer.json`)

### Purpose

The tokenizer converts between human-readable text and token IDs that the model understands. It handles:
- **Encoding**: Text → Token IDs
- **Decoding**: Token IDs → Text
- **Special Tokens**: BOS (beginning of sequence), EOS (end of sequence), padding, etc.

---

## File 4: Embedding Model (`embed_tokens.onnx`) - CRITICAL FOR PROPER GENERATION

### Purpose

The embedding model converts token IDs to dense vector embeddings. This is **essential** for properly combining image and text inputs in the first forward pass.

**Why It's Critical**: Without this model, we cannot convert question token IDs to embeddings, making it impossible to properly conditionally merge image embeddings with question embeddings (replacing the `<image>` token). This leads to nonsensical outputs like "rowsiness" because the model doesn't receive the question context.

### Research Findings

According to Hugging Face documentation and the SmolVLM-256M-Instruct README:
- The `embed_tokens.onnx` model is the **official way** to convert token IDs to embeddings
- It ensures embeddings are consistent with the model's training phase
- It must be used to properly conditionally merge image and text embeddings (replacing the `<image>` token)

### Input Requirements

**Input: `input_ids`**
- **Type**: `int64` tensor (BigInt64Array)
- **Shape**: `[batch, sequence_length]`
- **Content**: Token IDs from the tokenizer

### Output

**Output: Token Embeddings**
- **Type**: `float32` tensor
- **Shape**: `[batch, sequence_length, embedding_dim]`
- **Purpose**: Dense vector representations of tokens that can be conditionally merged with image embeddings (replacing the `<image>` token)

### How We Interface

```typescript
// 1. Load embed_tokens.onnx (optional, fails gracefully if not available)
const embedTokensUrl = `${SMOLVLM_MODEL_BASE_URL}/embed_tokens.onnx`;
const embedTokensResponse = await fetchWithProgress(embedTokensUrl, ...);
const embedTokensArrayBuffer = await embedTokensResponse.arrayBuffer();
smolvlmModel.embedTokens = await ort.InferenceSession.create(embedTokensArrayBuffer, ...);

// 2. Format prompt with <image> token placeholder
const prompt = `<|im_start|>User: <image> ${question}<end_of_utterance>\nAssistant:`;
const questionTokenIds = tokenizeText256M(prompt);

// 3. Find the <image> token ID and its index in the tokenized sequence
const imageTokenId = smolvlmModel.tokenizer.encode('<image>').ids[0];
const imageTokenIndex = questionTokenIds.indexOf(imageTokenId);

// 4. Convert question token IDs to embeddings (includes <image> token)
const inputIdsTensor = new ort.Tensor(
  'int64',
  new BigInt64Array(questionTokenIds.map(id => BigInt(id))),
  [1, questionTokenIds.length]
);
const embedOutputs = await smolvlmModel.embedTokens.run({ input_ids: inputIdsTensor });
const questionEmbeddings = embedOutputs[Object.keys(embedOutputs)[0]];

// 5. Conditional merge: Replace <image> token embedding with image embeddings
// This is the CRITICAL step - we don't concatenate, we replace!
const questionSeqLen = questionEmbeddings.dims[1]; // Full prompt length (includes <image> token)
const imageSeqLen = imageEmbeddings.dims[1]; // ~64 for 512x512 image
const totalSeqLen = (questionSeqLen - 1) + imageSeqLen; // Replace 1 token with imageSeqLen tokens

const mergedData = new Float32Array(imageBatch * totalSeqLen * embeddingDim);

// Copy question embeddings before <image> token
mergedData.set(
  questionEmbeddings.data.subarray(0, imageTokenIndex * embeddingDim),
  0
);

// Replace <image> token with image embeddings
mergedData.set(
  imageEmbeddings.data,
  imageTokenIndex * embeddingDim
);

// Copy question embeddings after <image> token
const afterImageStart = (imageTokenIndex + 1) * embeddingDim;
const afterImageLen = (questionSeqLen - imageTokenIndex - 1) * embeddingDim;
mergedData.set(
  questionEmbeddings.data.subarray(afterImageStart, afterImageStart + afterImageLen),
  imageTokenIndex * embeddingDim + imageSeqLen * embeddingDim
);

const mergedEmbeddings = new ort.Tensor(
  'float32',
  mergedData,
  [imageBatch, totalSeqLen, embeddingDim]
);

// 6. Pass to decoder as inputs_embeds
decoderInputs = {
  inputs_embeds: mergedEmbeddings,
  attention_mask: ...,
  position_ids: ...,
  ...pastKeyValueInputs
};
```

### Key Challenges

1. **File Availability**: The `embed_tokens.onnx` file may not always be available in the repository. Our code handles this gracefully by falling back to other approaches.

2. **Conditional Merge (NOT Concatenation)**: Image embeddings must **replace** the `<image>` token's embedding within the tokenized text sequence. This is a 1-to-N replacement (1 token → ~64 image patch embeddings), not a simple concatenation.

3. **Image Token Placeholder**: The `<image>` token **MUST** be included in the prompt and tokenized. Its embedding position is then replaced by the vision encoder's output.

4. **Shape Matching**: The embedding dimensions must match between image and question embeddings for proper replacement. The final sequence length is `(questionSeqLen - 1) + imageSeqLen`.

### Research Sources

- [SmolVLM-256M-Instruct README](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/README.md) - Official sample code
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/model_doc/smolvlm) - Model architecture details

---

## File 3: Tokenizer (`tokenizer.json`)

### File Format

- **Format**: JSON file containing tokenizer configuration
- **Library**: `@huggingface/tokenizers` (JavaScript implementation)
- **Size**: ~3.5MB

### How We Interface

```typescript
// 1. Download tokenizer.json
const tokenizerResponse = await fetch(tokenizerUrl);
const tokenizerArrayBuffer = await tokenizerResponse.arrayBuffer();
const tokenizerJson = JSON.parse(new TextDecoder().decode(tokenizerArrayBuffer));

// 2. Initialize tokenizer
import { Tokenizer } from '@huggingface/tokenizers';
const tokenizer = new Tokenizer(tokenizerJson, tokenizerJson);

// 3. Encode text (text → token IDs)
const encoded = tokenizer.encode("What is in this image?");
const tokenIds = encoded.ids; // [1234, 5678, ...]

// 4. Decode tokens (token IDs → text)
const decoded = tokenizer.decode([1234, 5678, ...]);
// "What is in this image?"
```

### Chat Template

SmolVLM uses a specific chat template format for questions. Based on the official Hugging Face implementation and technical guidance, the correct format is:

```typescript
function formatVQAPrompt256M(question: string): string {
  // Format: "<|im_start|>User: <image> {question}<end_of_utterance>\nAssistant:"
  // The <image> token will be replaced by image embeddings during the merge step
  return `<|im_start|>User: <image> ${question}<end_of_utterance>\nAssistant:`;
}
```

For captioning (empty question), we use:
```typescript
const prompt = question.length > 0 
  ? formatVQAPrompt256M(question)
  : formatVQAPrompt256M('Can you describe this image?');
```

**Critical Components:**
- **`<|im_start|>`**: Beginning of sequence (BOS) token - marks the start of the conversation
- **`User:`**: Marks the start of the user's turn
- **`<image>`**: **CRITICAL** - This token placeholder will be replaced by image embeddings during the conditional merge step
- **`<end_of_utterance>`**: Marks the end of a user or assistant turn
- **`Assistant:`**: The final string that prompts the model to begin generating its response

**Important Notes:**
- The `<image>` token **MUST** be included in the prompt - it's not optional
- The `<image>` token's embedding will be replaced by the vision encoder's output during the merge step
- The model was trained with this exact format, so deviating from it causes garbage output

### Key Challenges

1. **Tokenizer Location**: The `tokenizer.json` file is in the root directory, not in the `onnx/` subdirectory.

2. **Library Version**: `@huggingface/tokenizers` version `^0.0.6` uses a constructor-based API, not `fromJSON()`.

3. **Type Safety**: The tokenizer JSON must be validated as an object before use.

---

## ViT-GPT2: Image Captioning with Transformers.js

### Overview

ViT-GPT2 is a vision-language model that combines a Vision Transformer (ViT) encoder with a GPT-2 decoder for image captioning. Unlike SmolVLM, which requires manual ONNX model management, ViT-GPT2 uses Transformers.js, which handles model loading, tokenization, and inference automatically.

**Model**: `Xenova/vit-gpt2-image-captioning`
**Endpoint**: `/image-captioning`
**Library**: `@xenova/transformers`

### Architecture

- **Vision Encoder**: ViT (Vision Transformer) processes images into embeddings
- **Text Decoder**: GPT-2 generates captions from image embeddings
- **Pipeline**: Transformers.js provides a unified `image-to-text` pipeline

### How We Interface

```typescript
import { pipeline, type Pipeline } from '@xenova/transformers';

// Initialize pipeline (handles model loading automatically)
const imageToTextPipeline = await pipeline(
  'image-to-text',
  'Xenova/vit-gpt2-image-captioning',
  {
    progress_callback: (progress) => {
      // Track download progress
    }
  }
);

// Generate caption from image (data URL or ImageData)
const imageDataUrl = canvas.toDataURL('image/png');
const result = await imageToTextPipeline(imageDataUrl);

// Extract caption text
const caption = Array.isArray(result) && result.length > 0
  ? (typeof result[0] === 'object' && result[0] !== null && 'generated_text' in result[0]
      ? String(result[0].generated_text)
      : '')
  : '';
```

### Key Advantages

1. **Automatic Model Management**: Transformers.js handles downloading, caching, and loading ONNX models
2. **Simplified API**: Single pipeline call replaces manual tensor management
3. **Built-in Tokenization**: No need to manually handle tokenizers
4. **CORS Proxy Support**: Custom fetch function handles Hugging Face CDN restrictions

### Differences from SmolVLM

| Feature | SmolVLM | ViT-GPT2 |
|---------|---------|----------|
| Model Format | Manual ONNX files | Transformers.js pipeline |
| Tensor Management | Manual | Automatic |
| Tokenization | Manual (`@huggingface/tokenizers`) | Built-in |
| File Downloads | Manual (vision_encoder.onnx, decoder.onnx, etc.) | Automatic via Transformers.js |
| Complexity | High (tensor shapes, past_key_values, etc.) | Low (simple pipeline call) |
| Control | Full control over inference | Limited to pipeline API |

### CORS Proxy Integration

Transformers.js uses a custom fetch function to handle CORS restrictions:

```typescript
// Override Transformers.js fetch to use CORS proxies
env.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
  return customFetch(input, init, onLog);
};
```

This ensures models can be loaded from Hugging Face even when direct requests are blocked.

---

## Function Calling Agent: DistilGPT-2 with WASM Tools

### Overview

The Function Calling Agent demonstrates client-side autonomous agents with local LLM inference. It uses DistilGPT-2 (a small, efficient text generation model) combined with Rust WASM tools to enable function calling capabilities.

**Model**: `Xenova/distilgpt2` (DistilGPT-2)
**Endpoint**: `/function-calling`
**Library**: `@xenova/transformers` + Rust WASM (`wasm-agent-tools`)

### Architecture

```
User Goal → LLM (DistilGPT-2) → Function Call Parsing → WASM Tool Execution → Result → Final Answer
```

1. **LLM Inference**: DistilGPT-2 generates text based on the goal
2. **Function Call Parsing**: Extract function calls from LLM output
3. **WASM Tool Execution**: Execute tools via Rust WASM module
4. **Result Integration**: Feed tool results back to LLM for final answer

### WASM Agent Tools Module

The `wasm-agent-tools` Rust crate provides three tools:

#### 1. `calculate(expression: &str) -> Result<String, JsValue>`

Evaluates mathematical expressions:
- Supports: `+`, `-`, `*`, `/`, parentheses
- Returns: Result as string
- Example: `calculate("2 + 2 * 3")` → `"8"`

#### 2. `process_text(text: &str, operation: &str) -> Result<String, JsValue>`

Processes text with various operations:
- `uppercase`: Convert to uppercase
- `lowercase`: Convert to lowercase
- `reverse`: Reverse character order
- `length`: Return character count
- `word_count`: Return word count
- Example: `process_text("Hello World", "uppercase")` → `"HELLO WORLD"`

#### 3. `get_stats(data: &[u8]) -> Result<String, JsValue>`

Calculates statistics from byte array:
- Returns JSON: `{"count": N, "min": M, "max": X, "sum": S, "average": A}`
- Example: `get_stats([1, 2, 3, 4, 5])` → `{"count":5,"min":1,"max":5,"sum":15,"average":3.00}`

### How We Interface

```typescript
// 1. Load WASM agent tools module
import initWasm from '../../pkg/wasm_agent_tools/wasm_agent_tools.js';
const wasmModuleExports = await initWasm();
const wasmModule = validateAgentToolsModule(wasmModuleExports);

// 2. Load LLM model
import { loadAgentModel, runAgent } from '../models/function-calling';
await loadAgentModel(onProgress, onLog);

// 3. Execute tool function
const executeTool = async (functionName: string, args: Record<string, string>): Promise<string> => {
  switch (functionName) {
    case 'calculate':
      return wasmModule.calculate(args.expression);
    case 'process_text':
      return wasmModule.process_text(args.text, args.operation);
    case 'get_stats':
      const dataArray = parseArrayString(args.data);
      const dataUint8 = new Uint8Array(dataArray);
      return wasmModule.get_stats(dataUint8);
    default:
      throw new Error(`Unknown function: ${functionName}`);
  }
};

// 4. Run agent with goal
const steps = await runAgent(
  goal,
  executeTool,
  onLog,
  5, // max iterations
  onClarify // optional clarification callback
);
```

### Agent Loop

The agent runs in a loop until a final answer is obtained:

```typescript
for (let step = 1; step <= maxIterations; step++) {
  // 1. Generate LLM output
  const llmOutput = await textGenerationPipeline(conversationHistory, {
    max_new_tokens: 30,
    temperature: 0.2,
    top_p: 0.4,
    repetition_penalty: 1.5
  });
  
  // 2. Parse function call from output
  const functionCall = parseFunctionCall(llmOutput);
  
  // 3. Execute function if found
  if (functionCall) {
    const result = await executeTool(functionCall.function, functionCall.arguments);
    conversationHistory += `\nFunction: ${functionCall.function}(${functionCall.arguments})\nResult: ${result}`;
  }
  
  // 4. Check for final answer
  if (hasFinalAnswer(llmOutput, functionCall)) {
    break;
  }
}
```

### Prompt Engineering

The agent uses dynamic prompts based on goal type:

**Math Goals** (e.g., "What is 2 + 2?"):
```
Goal: 2 + 2

Step 1: Call calculate(expression="2 + 2")
Step 2: Output the result as the final answer.
```

**Array Goals** (e.g., "[1, 2, 3]"):
```
Goal: [1, 2, 3]

Step 1: Call get_stats(data="[1, 2, 3]")
Step 2: Output the result as the final answer.
```

**Text Goals** (e.g., "Make uppercase: hello"):
```
Goal: Make uppercase: hello

Use process_text function. Available operations: uppercase, lowercase, reverse, length, word_count
Call process_text with text="hello" and choose appropriate operation.
```

### Key Features

1. **Goal Type Detection**: Automatically detects math, array, or text goals
2. **Direct Execution**: For inferred operations, directly executes WASM tools without LLM
3. **Context Window Management**: Truncates conversation history to fit DistilGPT-2's 1024-token limit
4. **Output Cleaning**: Aggressively filters garbage output from base GPT-2 model
5. **Function Call Validation**: Only executes known functions (`calculate`, `process_text`, `get_stats`)

### Model Choice: DistilGPT-2

**Why DistilGPT-2?**
- **Small Size**: ~350MB, fits in browser memory
- **Fast Inference**: Quick generation for interactive use
- **Proven Compatibility**: Works reliably with Transformers.js
- **Context Window**: 1024 tokens (sufficient for simple function calling)

**Limitations:**
- Base model (not instruction-tuned), requires aggressive prompt engineering
- Limited reasoning capabilities
- May generate repetitive or nonsensical output without careful prompting

---

## The Download Journey

### CORS Proxy System

Hugging Face CDN doesn't allow direct cross-origin requests from browsers. We use a fallback chain of CORS proxy services:

1. **`api.allorigins.win/raw?url=`** - Primary proxy
2. **`corsproxy.io/?`** - Secondary proxy
3. **`api.codetabs.com/v1/proxy?quest=`** - Tertiary proxy
4. **`cors-anywhere.herokuapp.com/`** - Fallback (may be rate-limited)
5. **`whateverorigin.org/get?url=`** - Last resort (returns JSON-wrapped content)

**Proxy Selection Logic:**
- Try each proxy in order
- Skip proxies returning error status codes (403, 408, 500, 502, 503, 504, redirects)
- Validate responses (check for HTML error pages, suspiciously small files)
- Fall back to direct fetch if all proxies fail

### Caching System

We use the browser's Cache API to store downloaded models:

- **Cache Name**: `smolvlm-models-v1`
- **Strategy**: Check cache before download, save to cache after successful download
- **Benefits**: Faster subsequent loads, reduced bandwidth usage
- **User Control**: "Clear Cache" button to reset downloads

### Progress Tracking

We track download progress with:
- **Percentage**: When `Content-Length` header is available
- **Bytes Loaded**: When size is unknown (estimates based on typical file sizes)
- **Timeout Detection**: 30 seconds of no data triggers an error
- **Periodic Updates**: Progress reported every 2 seconds or every 1MB

### Error Handling

- **HTML Detection**: Validates that responses are binary, not HTML error pages
- **Size Validation**: Ensures files are reasonably large (ONNX models are several MB)
- **Protobuf Validation**: ONNX Runtime validates file format during session creation
- **User-Friendly Messages**: Clear error messages with troubleshooting suggestions

---

## The Inference Pipeline

### Complete Flow

1. **Image Upload/Webcam Capture**
   - User provides image via file input or webcam

2. **WASM Preprocessing**
   - Decode image (PNG/JPEG)
   - Center crop to square
   - Resize to target size (512×512 for 256M, 224×224 for 500M)
   - Convert RGBA → RGB
   - Normalize pixels to `[0, 1]` range
   - Return `Float32Array`

3. **Vision Encoder**
   - Reshape image data to 5D tensor
   - Create `pixel_attention_mask`
   - Run vision encoder
   - Extract image embeddings

4. **Question Formatting**
   - Format question with chat template: `<|im_start|>User: <image> {question}<end_of_utterance>\nAssistant:`
   - Tokenize question text (includes `<image>` token ID)
   - Find `<image>` token index in tokenized sequence

5. **Conditional Merge**
   - Get embeddings for full tokenized sequence (including `<image>` token)
   - Replace `<image>` token's embedding with image embeddings sequence
   - Final sequence length: `(questionSeqLen - 1) + imageSeqLen`

6. **Decoder Initialization**
   - Prepare initial decoder inputs with merged embeddings
   - Initialize empty `past_key_values`

6. **Autoregressive Generation**
   - Loop for up to `MAX_GENERATION_LENGTH` steps:
     - Run decoder
     - Extract logits
     - Get next token (argmax)
     - Check for EOS token
     - Update `past_key_values`
     - Prepare inputs for next iteration

8. **Text Decoding**
   - Decode generated token IDs to text
   - Return final answer

---

## Tensor Shapes and Type Safety

We use TypeScript type aliases to ensure tensor shape correctness:

```typescript
type PastKeyValueShape = [number, number, number, number]; // [batch, num_heads, seq_len, head_dim]
type ImageTensorShape = [number, number, number, number, number]; // [batch, num_images, channels, height, width]
type PixelAttentionMaskShape = [number, number, number, number]; // [batch, num_images, height, width]
type DecoderAttentionMaskShape = [number, number]; // [batch, sequence_length]
type PositionIdsShape = [number, number]; // [batch, sequence_length]
type InputIdsShape = [number, number]; // [batch, sequence_length]
```

These types prevent rank mismatches and dimension errors at compile time.

---

## Known Challenges and Solutions

### Challenge 1: Token Embeddings Without Embedding Layer ✅ SOLVED

**Problem**: For autoregressive generation, we need embeddings for new tokens. Without access to the embedding layer, we can't convert token IDs to embeddings.

**Solution**: Use `embed_tokens.onnx` model (if available) to convert token IDs to embeddings. This is the official approach recommended by Hugging Face and documented in the SmolVLM-256M-Instruct README.

**Implementation**:
1. Load `embed_tokens.onnx` during model initialization (optional, fails gracefully if not available)
2. Format prompt with `<image>` token placeholder: `<|im_start|>User: <image> {question}<end_of_utterance>\nAssistant:`
3. Tokenize prompt (includes `<image>` token ID)
4. Find `<image>` token index in tokenized sequence
5. For first forward pass: Convert question token IDs (including `<image>`) to embeddings using `embed_tokens.onnx`
6. **Conditional merge**: Replace `<image>` token's embedding with image embeddings (1 token → ~64 image patch embeddings)
7. Pass merged embeddings as `inputs_embeds` to decoder
8. For subsequent tokens: Use `embed_tokens.onnx` to convert token ID to embedding, then pass to decoder

**Research Source**: [SmolVLM-256M-Instruct README](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/README.md)

### Challenge 2: Mixing Image and Text Inputs ✅ SOLVED

**Problem**: We need to provide both image embeddings and question tokens in the first forward pass, but ONNX models typically don't allow mixing `input_ids` and `inputs_embeds`.

**Solution**: 
- **Correct Approach**: Use `embed_tokens.onnx` to convert question token IDs (including `<image>` token) to embeddings, then **conditionally replace** the `<image>` token's embedding with image embeddings
- This is a **conditional merge**, not a simple concatenation
- The `<image>` token in the prompt gets replaced by the vision encoder's output (1 token → ~64 image patch embeddings)
- Final sequence: `[tokens_before_image, image_embeds, tokens_after_image]`

**Research Finding**: According to technical guidance and Hugging Face's `SmolVLMModel.inputs_merger` implementation, the proper approach is:
```python
# Simplified logic from SmolVLMModel.inputs_merger
# image_mask is a boolean tensor where True marks the position of the <image> token(s)
merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
```

This replaces the `<image>` token embedding with image embeddings, not concatenates them.

### Challenge 3: Past Key Values Extraction

**Problem**: `past_key_values` must be correctly extracted from decoder outputs and passed to the next iteration. Missing or incorrect shapes cause errors.

**Solution**: 
- Iterate through all decoder output keys
- Extract all `past_key_values.*` tensors
- Ensure all required `past_key_values` inputs are present (reuse previous values or create empty tensors if missing)

### Challenge 4: CORS and Proxy Reliability

**Problem**: Hugging Face CDN blocks direct browser requests. CORS proxies are unreliable.

**Solution**: 
- Multiple proxy fallback chain
- Robust error detection and retry logic
- Direct fetch as last resort
- Caching to reduce dependency on proxies

---

## Future Improvements

### 1. Embedding Layer Extraction

Extract embedding weights from the ONNX decoder model to enable proper token embedding conversion. This would allow us to:
- Conditionally merge image embeddings with question embeddings in `inputs_embeds` (replacing `<image>` token)
- Use proper embeddings for new tokens in autoregressive generation

### 2. Model Input Structure Analysis

Create a tool to automatically analyze ONNX model inputs/outputs and generate TypeScript interfaces. This would:
- Reduce manual errors
- Improve type safety
- Make it easier to support new models

### 3. Improved Error Messages

Provide more specific error messages based on common failure modes:
- "Model expects 5D tensor but got 4D" → Show expected vs actual shape
- "Missing input: past_key_values.0.key" → List all required inputs
- "Invalid token embedding" → Suggest using `input_ids` if supported

### 4. Generation Quality Improvements

- **Temperature Sampling**: Instead of argmax, use temperature-based sampling for more diverse outputs
- **Top-k/Top-p Sampling**: Limit sampling to top-k tokens or nucleus (top-p)
- **Beam Search**: Generate multiple candidates and select the best
- **Repetition Penalty**: Reduce repetitive token generation

### 5. Performance Optimizations

- **WebGPU Backend**: Use WebGPU execution provider for faster inference (if available)
- **Model Quantization**: Further optimize with INT4 quantization (if available)
- **Streaming Generation**: Stream tokens as they're generated (better UX)

### 6. Model Variant Support

- **SmolVLM-1B**: Support for larger 1B parameter model
- **Fine-tuned Variants**: Support for domain-specific fine-tuned models

---

## Troubleshooting Garbage Output

If the model generates repetitive garbage output (e.g., "sersymour refund laptigALTH livejoice..."), check:

1. **Prompt Format**: Ensure the prompt format is `<|im_start|>User: <image> {question}<end_of_utterance>\nAssistant:` with the `<image>` token included
2. **Conditional Merge**: Verify that image embeddings **replace** the `<image>` token's embedding, not concatenate with it
3. **Image Token Index**: Confirm the `<image>` token is found in the tokenized sequence and its index is correct
4. **Sequence Length**: Verify the final sequence length is `(questionSeqLen - 1) + imageSeqLen` (replacing 1 token with imageSeqLen tokens)
5. **Position IDs**: Verify position_ids are calculated as `initialSequenceLength + generatedTokenIds.length` for subsequent iterations
6. **Repetition Detection**: Check that long pattern detection (5-gram, 10-gram, sliding window) is working

Common issues:
- **Missing `<image>` token**: The prompt must include `<image>` token placeholder - without it, the model can't properly merge image and text
- **Incorrect merge**: Concatenating `[image_embeds, question_embeds]` instead of replacing the `<image>` token causes garbage output from the first token
- **Wrong sequence length**: Using `imageSeqLen + questionSeqLen` instead of `(questionSeqLen - 1) + imageSeqLen` causes dimension mismatches
- **Wrong position IDs**: Using `currentSequencePosition - 1` instead of absolute position causes misalignment
- **Insufficient repetition detection**: Only checking 2-gram/3-gram misses longer patterns (10-12 tokens)

---

## Testing Notes

After running tests, update this section with:
- Actual file sizes observed
- Successful proxy services
- Common error patterns
- Performance metrics
- Generation quality observations

---

## References

### SmolVLM
- **Hugging Face Models**:
  - [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
  - [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- **ONNX Runtime Web**: [Documentation](https://onnxruntime.ai/docs/tutorials/web/)
- **Hugging Face Tokenizers**: [Documentation](https://github.com/huggingface/tokenizers)

### ViT-GPT2
- **Model**: [Xenova/vit-gpt2-image-captioning](https://huggingface.co/Xenova/vit-gpt2-image-captioning)
- **Transformers.js**: [Documentation](https://huggingface.co/docs/transformers.js/)

### Function Calling Agent
- **Model**: [Xenova/distilgpt2](https://huggingface.co/Xenova/distilgpt2)
- **Transformers.js**: [Documentation](https://huggingface.co/docs/transformers.js/)
- **DistilGPT-2 Paper**: [DistilBERT: a distilled version of BERT](https://arxiv.org/abs/1910.01108)

---

*Last Updated: [Date will be updated after tests]*

