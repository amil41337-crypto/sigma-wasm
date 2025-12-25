# Learning From Mistakes: A Comprehensive Guide to Client-Side AI Development

This document captures critical lessons learned from building a production web application with client-side LLMs, WASM modules, and complex deployment pipelines. Every mistake documented here was a real production issue that caused failures, and every solution was hard-won through debugging and research.

## Table of Contents

1. [Overview](#overview)
2. [Critical Production Deployment Mistakes](#critical-production-deployment-mistakes)
   - [The render.yaml Build Filter Disaster](#the-renderyaml-build-filter-disaster)
   - [Error Handling Anti-Patterns](#error-handling-anti-patterns)
   - [Error Logging Best Practices](#error-logging-best-practices)
3. [LLM Integration Learnings](#llm-integration-learnings)
   - [SmolVLM: Vision-Language Models](#smolvlm-vision-language-models)
   - [ViT-GPT2: Image Captioning with Transformers.js](#vit-gpt2-image-captioning-with-transformersjs)
   - [Function Calling Agent: DistilGPT-2 with WASM Tools](#function-calling-agent-distilgpt-2-with-wasm-tools)
4. [The Download Journey](#the-download-journey)
5. [The Inference Pipeline](#the-inference-pipeline)
6. [Tensor Shapes and Type Safety](#tensor-shapes-and-type-safety)
7. [Known Challenges and Solutions](#known-challenges-and-solutions)
8. [Production Deployment Checklist](#production-deployment-checklist)
9. [Future Improvements](#future-improvements)

---

## Overview

This application integrates multiple LLM approaches for different use cases, but more importantly, it documents the mistakes we made and how we fixed them. Each section below includes:

- **What We Learned**: The key takeaway
- **The Mistake**: What went wrong
- **The Impact**: How it affected production
- **The Solution**: How we fixed it

### LLM Approaches

1. **SmolVLM (ONNX Runtime Web)**: Vision-Language Models for image understanding
   - SmolVLM-500M: 500 million parameters, uses 224×224 images
   - SmolVLM-256M: 256 million parameters, uses 512×512 images
   - Endpoints: `/preprocess-smolvlm-500m`, `/preprocess-smolvlm-256m`

2. **ViT-GPT2 (Transformers.js)**: Image Captioning Model
   - Model: `Xenova/vit-gpt2-image-captioning`
   - Endpoint: `/image-captioning`

3. **Function Calling Agent (Transformers.js + WASM)**: Autonomous Agent
   - Model: `Xenova/distilgpt2` (DistilGPT-2)
   - Tools: WASM-based tools (`calculate`, `process_text`, `get_stats`)
   - Endpoint: `/function-calling`

---

## Critical Production Deployment Mistakes

### The render.yaml Build Filter Disaster

**What We Learned**: Always verify build configuration files include ALL required source directories. Missing directories in build filters cause silent failures in production.

**The Mistake**: The `render.yaml` file's `buildFilter.paths` section was missing three WASM module directories:
- `wasm-preprocess-256m/**`
- `wasm-preprocess-image-captioning/**`
- `wasm-agent-tools/**`

**The Impact**: 
- All WASM modules failed to load in production with generic "Failed to load WASM module" errors
- The Docker build succeeded, but the WASM source code wasn't included in the build context
- No error during build - the missing directories were silently excluded
- Users saw broken functionality on all endpoints using these modules

**The Root Cause**: 
- When adding new WASM modules, we updated the `Dockerfile` and `Cargo.toml` but forgot to update `render.yaml`
- Render.com's build filter excludes everything not explicitly listed in `paths`
- The build appeared successful because Docker didn't fail - it just didn't have the source files

**The Solution**:
```yaml
buildFilter:
  paths:
    - src/**
    - wasm-astar/**
    - wasm-preprocess/**
    - wasm-preprocess-256m/**              # ADDED
    - wasm-preprocess-image-captioning/**  # ADDED
    - wasm-agent-tools/**                  # ADDED
    - Cargo.toml
    # ... rest of paths
```

**Key Lesson**: When adding new modules or directories:
1. Update `Dockerfile` (source copying)
2. Update `Cargo.toml` (workspace members)
3. Update `render.yaml` (build filter paths) ← **EASY TO FORGET**
4. Update `scripts/build.sh` (build script)
5. Update `vite.config.ts` (if needed for routing)

**Prevention**: Create a checklist for adding new WASM modules (see [Production Deployment Checklist](#production-deployment-checklist))

---

### Error Handling Anti-Patterns

**What We Learned**: Generic error messages that don't preserve original error details make production debugging impossible. Always include the original error message when wrapping errors.

**The Mistake**: The `loadWasmModule` function wrapped errors in a generic message:

```typescript
// BAD: Loses original error message
catch (error) {
  throw new WasmLoadError('Failed to load WASM module', error);
}
```

**The Impact**:
- Production errors showed only "Failed to load WASM module: Failed to load WASM module"
- No way to know if it was a network error, file not found, or initialization failure
- Debugging required guessing what the actual error might be
- Users saw unhelpful error messages

**The Root Cause**:
- Error wrapping pattern didn't extract the original error message
- The `cause` property existed but wasn't displayed to users
- Error messages were too generic to be actionable

**The Solution**:
```typescript
// GOOD: Preserves original error message
catch (error) {
  if (error instanceof WasmInitError) {
    throw error;
  }
  const errorMessage = error instanceof Error 
    ? error.message 
    : String(error);
  throw new WasmLoadError(`Failed to load WASM module: ${errorMessage}`, error);
}
```

**Key Lesson**: 
- Always extract and include the original error message in wrapped errors
- Use template strings to combine context with original message
- Preserve the original error as the `cause` for stack traces

**Prevention**: 
- Use a lint rule to catch error wrapping without message extraction
- Always test error messages in production-like environments
- Include error message extraction in code review checklist

---

### Error Logging Best Practices

**What We Learned**: Comprehensive error logging with stack traces, import paths, and error causes is essential for production debugging. Generic logs are useless.

**The Mistake**: Error logging was minimal:
- Only logged generic error messages
- No stack traces
- No import paths
- No error causes

**The Impact**:
- Production debugging required reproducing issues locally
- No way to diagnose issues from logs alone
- Had to guess what import path was failing
- Couldn't see the full error chain

**The Solution**: Enhanced error logging in all route files:

```typescript
catch (error) {
  const errorMsg = error instanceof Error ? error.message : 'Unknown error';
  if (addLogEntry) {
    addLogEntry(`Failed to load WASM module: ${errorMsg}`, 'error');
    addLogEntry(`Import path: ../../pkg/wasm_agent_tools/wasm_agent_tools.js`, 'info');
    if (error instanceof Error && error.stack) {
      addLogEntry(`Error stack: ${error.stack}`, 'error');
    }
    if (error instanceof Error && 'cause' in error && error.cause) {
      const causeMsg = error.cause instanceof Error 
        ? error.cause.message 
        : typeof error.cause === 'string' 
          ? error.cause 
          : JSON.stringify(error.cause);
      addLogEntry(`Error cause: ${causeMsg}`, 'error');
    }
  }
  throw error;
}
```

**Key Lesson**:
- Log import paths being used (helps identify path resolution issues)
- Log full stack traces (shows where errors originate)
- Log error causes (shows the full error chain)
- Use proper type narrowing for error causes (avoid `[object Object]`)

**Prevention**:
- Create a standard error logging pattern for all routes
- Include error logging in code review checklist
- Test error logging in production-like environments

---

## LLM Integration Learnings

### SmolVLM: Vision-Language Models

**What We Learned**: Vision-language models require careful tensor shape management, proper embedding merging, and understanding of autoregressive generation patterns.

#### The Three Essential Files (Plus One Critical)

To run SmolVLM in the browser, we need three essential files downloaded from Hugging Face, plus one critical file for proper text embedding:

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

4. **`embed_tokens.onnx`** (CRITICAL, ~50-100MB)
   - **CRITICAL for proper text generation**: Converts token IDs to embeddings
   - Allows proper conditional merge of image embeddings with question embeddings (replacing `<image>` token)
   - Location: `{MODEL_BASE_URL}/onnx/embed_tokens.onnx`
   - **Without this file**: The model cannot properly combine image and text inputs, leading to nonsensical outputs

**Base URLs:**
- 500M: `https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/resolve/main`
- 256M: `https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main`

#### Key Challenges We Overcame

**Challenge 1: 5D Tensor Requirement**
- **Mistake**: Initially tried to use 4D tensors `[batch, channels, height, width]`
- **Reality**: ONNX expects `[batch, num_images, channels, height, width]` (5D)
- **Solution**: Add the `num_images` dimension (always `1` for single images)

**Challenge 2: Conditional Merge (NOT Concatenation)**
- **Mistake**: Initially concatenated image embeddings with question embeddings
- **Reality**: Must **replace** the `<image>` token's embedding with image embeddings
- **Solution**: Find `<image>` token index, replace its embedding with image embeddings sequence
- **Critical**: This is a 1-to-N replacement (1 token → ~64 image patch embeddings)

**Challenge 3: Token Embeddings Without Embedding Layer**
- **Mistake**: Tried to use decoder's internal embedding layer (not accessible in ONNX)
- **Reality**: Need `embed_tokens.onnx` to convert token IDs to embeddings
- **Solution**: Load `embed_tokens.onnx` separately and use it for token→embedding conversion

See the original detailed sections below for complete implementation details.

---

### ViT-GPT2: Image Captioning with Transformers.js

**What We Learned**: Transformers.js dramatically simplifies model management compared to manual ONNX handling, but requires understanding of pipeline types and proper input formats.

**Model**: `Xenova/vit-gpt2-image-captioning`
**Endpoint**: `/image-captioning`
**Library**: `@xenova/transformers`

#### Key Advantages

1. **Automatic Model Management**: Transformers.js handles downloading, caching, and loading ONNX models
2. **Simplified API**: Single pipeline call replaces manual tensor management
3. **Built-in Tokenization**: No need to manually handle tokenizers
4. **CORS Proxy Support**: Custom fetch function handles Hugging Face CDN restrictions

#### Input Format Mistake

**Mistake**: Initially tried to pass `ImageData` or `HTMLCanvasElement` directly
**Reality**: Transformers.js expects data URL strings for image inputs
**Solution**: Convert canvas to data URL: `canvas.toDataURL('image/png')`

```typescript
// CORRECT: Use data URL
const dataUrl = canvas.toDataURL('image/png');
const result = await imageToTextPipeline(dataUrl);
```

---

### Function Calling Agent: DistilGPT-2 with WASM Tools

**What We Learned**: Base language models (not instruction-tuned) require aggressive prompt engineering and output cleaning. Function calling with small models is possible but requires careful design.

**Model**: `Xenova/distilgpt2` (DistilGPT-2)
**Endpoint**: `/function-calling`
**Library**: `@xenova/transformers` + Rust WASM (`wasm-agent-tools`)

#### Model Choice Lessons

**Why DistilGPT-2?**
- Small size (~350MB), fits in browser memory
- Fast inference for interactive use
- Proven compatibility with Transformers.js
- Context window: 1024 tokens (sufficient for simple function calling)

**Limitations We Encountered:**
- Base model (not instruction-tuned), requires aggressive prompt engineering
- Limited reasoning capabilities
- Generates repetitive or nonsensical output without careful prompting
- Requires extensive output cleaning to extract valid function calls

#### Prompt Engineering Mistakes

**Mistake 1**: Including hardcoded examples in prompts
- **Problem**: Model copied examples verbatim instead of using actual goal data
- **Solution**: Use dynamic prompts with actual goal data, no hardcoded examples

**Mistake 2**: Generic prompts for all goal types
- **Problem**: Model couldn't infer correct tool for different goal types
- **Solution**: Generate type-specific prompts (math, array, text) with explicit instructions

**Mistake 3**: Not guiding the model to produce final answers
- **Problem**: Model would call tools but never output final answer
- **Solution**: Explicitly instruct "Step 2: Output the result as the final answer"

#### Output Cleaning Lessons

**Mistake**: Trusting raw LLM output
- **Problem**: Base GPT-2 generates garbage (C++ code, repetitive patterns, etc.)
- **Solution**: Aggressive cleaning:
  - Extract only valid function calls
  - Filter out garbage text
  - Validate function names against known tools
  - Extract numbers or meaningful text from cleaned output

---

## The Download Journey

### CORS Proxy System

**What We Learned**: Hugging Face CDN doesn't allow direct cross-origin requests. Multiple proxy fallbacks are essential for reliability.

**Mistake**: Using a single CORS proxy
**Reality**: Proxies are unreliable and rate-limited
**Solution**: Fallback chain:
1. `api.allorigins.win/raw?url=` - Primary proxy
2. `corsproxy.io/?` - Secondary proxy
3. `api.codetabs.com/v1/proxy?quest=` - Tertiary proxy
4. `cors-anywhere.herokuapp.com/` - Fallback (may be rate-limited)
5. `whateverorigin.org/get?url=` - Last resort (returns JSON-wrapped content)

**Proxy Selection Logic:**
- Try each proxy in order
- Skip proxies returning error status codes (403, 408, 500, 502, 503, 504, redirects)
- Validate responses (check for HTML error pages, suspiciously small files)
- Fall back to direct fetch if all proxies fail

### Caching System

**What We Learned**: Browser Cache API is essential for large model files. Without caching, users re-download hundreds of MBs on every page load.

**Implementation:**
- Cache Name: `smolvlm-models-v1`
- Strategy: Check cache before download, save to cache after successful download
- Benefits: Faster subsequent loads, reduced bandwidth usage
- User Control: "Clear Cache" button to reset downloads

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

7. **Autoregressive Generation**
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

**What We Learned**: TypeScript type aliases for tensor shapes prevent rank mismatches and dimension errors at compile time.

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

## Production Deployment Checklist

Before deploying to production, verify:

### Build Configuration
- [ ] All WASM module directories in `render.yaml` `buildFilter.paths`
- [ ] All WASM modules in `Cargo.toml` workspace members
- [ ] All WASM modules in `Dockerfile` COPY commands
- [ ] All WASM modules in `scripts/build.sh`
- [ ] All routes in `vite.config.ts` `rollupOptions.input`
- [ ] All routes in `vite.config.ts` `devServerRouting` middleware
- [ ] All routes in `nginx.conf.template` location blocks

### Error Handling
- [ ] Error messages preserve original error details
- [ ] Error wrapping includes original message in new message
- [ ] Error causes are properly extracted and displayed
- [ ] Stack traces are logged for debugging

### Error Logging
- [ ] Import paths are logged
- [ ] Full error messages are logged
- [ ] Stack traces are logged
- [ ] Error causes are logged (when available)
- [ ] Type narrowing for error causes (avoid `[object Object]`)

### WASM Module Loading
- [ ] All WASM modules use `loadWasmModule` helper
- [ ] All WASM modules have proper validation functions
- [ ] WASM module paths are correct for both dev and production
- [ ] Vite configuration handles WASM correctly (`assetsInlineLimit: 0`)

### Testing
- [ ] Test all endpoints in production-like environment
- [ ] Verify WASM modules load correctly
- [ ] Verify error messages are helpful
- [ ] Verify error logging is comprehensive
- [ ] Test error scenarios (network failures, missing files, etc.)

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

*Last Updated: December 2024*

