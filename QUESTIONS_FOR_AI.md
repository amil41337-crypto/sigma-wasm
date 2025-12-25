# Questions for AI Assistants - SmolVLM-256M Integration

## Project Overview

This project integrates **SmolVLM-256M-Instruct** (a vision-language model) into a web application using:
- **ONNX Runtime Web** for model inference
- **Rust WASM** for image preprocessing
- **TypeScript** for the frontend
- **Vite** for building

The goal is to enable **image captioning** and **visual question answering (VQA)** in the browser.

## Current Problem

The model generates **repetitive garbage output** instead of coherent captions/answers. Examples:

- Captioning: `"iosync Debor tomatreponamefoliospected; ourselves meansALTH livejoice Herinsically about bodyweedspectymour refund laptigALTH livejoice..."` (repeats indefinitely)
- VQA: `"foundland herself with bodyweedspectymour refund laptigALTH livejoice Herinsically about bodyweedspectymour refund laptigALTH..."` (same pattern)

The repetition pattern is approximately **10-12 tokens long** and repeats continuously until hitting the max generation length (128 tokens).

## What We've Tried

### 1. Chat Template Format
- **Initial**: `<|user|>\n{question}\n<|assistant|>\n`
- **Updated to**: `User: {question}<end_of_utterance>\nAssistant:` (based on llama.cpp PR #13050)
- **Result**: Still generates garbage

### 2. Repetition Detection
- Added 2-gram, 3-gram, 5-gram, 10-gram pattern detection
- Added sliding window detection for patterns at any offset
- Added early pattern detection (5-token pattern appearing 3+ times)
- Added maximum repetition count checks
- Added low diversity checks
- **Result**: Detection works but doesn't prevent the initial garbage generation

### 3. Position IDs
- Fixed from `currentSequencePosition - 1` to `initialSequenceLength + generatedTokenIds.length`
- **Result**: Still generates garbage

### 4. EOS Token Detection
- Updated to check for `<end_of_utterance>` (SmolVLM format) and `<|endoftext|>` (fallback)
- **Result**: Model doesn't generate EOS tokens naturally

### 5. Image Embedding Concatenation
- Using `embed_tokens.onnx` to convert question token IDs to embeddings
- Concatenating image embeddings with question embeddings: `[image_embeds, question_embeds]`
- Shape: `[batch, imageSeqLen + questionSeqLen, embeddingDim]`
- **Result**: Still generates garbage

## Technical Details

### Model Files
- **Vision Encoder**: `vision_encoder.onnx` (~356 MB)
- **Decoder**: `decoder_model_merged_int8.onnx` (~130 MB, INT8 quantized)
- **Embedding Model**: `embed_tokens.onnx` (~108 MB) - converts token IDs to embeddings
- **Tokenizer**: `tokenizer.json` (~3 MB)

### Image Preprocessing
- **Target Size**: 512x512 pixels
- **Format**: Normalized Float32Array `[R, G, B, R, G, B, ...]`
- **Reshaped to**: `[1, 1, 3, 512, 512]` (5D tensor: batch, num_images, channels, height, width)
- **Vision Encoder Output**: Image embeddings with shape `[1, imageSeqLen, embeddingDim]` where `imageSeqLen ≈ 64` (for 512x512 image)

### Decoder Inputs
- **First Pass**: 
  - `inputs_embeds`: Concatenated `[image_embeds, question_embeds]` with shape `[1, imageSeqLen + questionSeqLen, 576]`
  - `attention_mask`: All ones, shape `[1, imageSeqLen + questionSeqLen]`
  - `position_ids`: Sequential `[0, 1, 2, ..., imageSeqLen + questionSeqLen - 1]`
  - `past_key_values.*`: Empty tensors with shape `[1, 3, 0, 64]` (batch, num_heads, seq_len=0, head_dim)
- **Subsequent Passes**:
  - `inputs_embeds`: Single token embedding from `embed_tokens.onnx` with shape `[1, 1, 576]`
  - `attention_mask`: All ones, shape `[1, 1]`
  - `position_ids`: `[initialSequenceLength + generatedTokenIds.length]`
  - `past_key_values.*`: Extracted from previous decoder outputs

### Logits Extraction
- **First Iteration**: Extract logits from last position: `logits[0, sequenceLength - 1, :]`
- **Subsequent Iterations**: Extract from position 0: `logits[0, 0, :]` (sequence length is 1)
- **Argmax**: Select token with highest probability

### Current Prompt Format
```typescript
// For captioning (empty question):
"User: Can you describe this image?<end_of_utterance>\nAssistant:"

// For VQA:
"User: {question}<end_of_utterance>\nAssistant:"
```

## Key Questions

### 1. Chat Template Format
**Question**: What is the **exact** chat template format that SmolVLM-256M-Instruct expects?

- We've tried `<|user|>\n{question}\n<|assistant|>\n` (Hugging Face format)
- We've tried `User: {question}<end_of_utterance>\nAssistant:` (llama.cpp format)
- **Neither works**. What is the correct format?

**Research Needed**:
- Check the actual `tokenizer_config.json` from Hugging Face
- Check if there's a `chat_template` field in the config
- Verify what `processor.apply_chat_template()` actually produces for SmolVLM-256M

### 2. Image Embedding Order
**Question**: Should image embeddings come **before** or **after** question embeddings in the concatenated sequence?

- **Current**: `[image_embeds, question_embeds]` (image first)
- **Alternative**: `[question_embeds, image_embeds]` (question first)
- Does the model expect a specific order?

**Research Needed**:
- Check Hugging Face's `SmolVLMProcessor` implementation
- Check how `processor(text=prompt, images=[image])` orders the embeddings
- Verify the training data format

### 3. Image Token Placeholder
**Question**: Does the text prompt need an **image token placeholder** that gets replaced by embeddings?

- Some models use `<image>` or `<|image|>` tokens in the prompt
- The llama.cpp PR mentions `<fake_token_around_image><global-img>{image_marker}<fake_token_around_image>` for IDEFICS3
- Should we include such tokens in the prompt?

**Research Needed**:
- Check if SmolVLM uses image tokens in the prompt
- Verify what `processor.apply_chat_template()` includes for image messages
- Check the model's tokenizer for image-related special tokens

### 4. Position IDs Calculation
**Question**: Are position IDs calculated correctly for the concatenated embeddings?

- **Current**: Position IDs are sequential `[0, 1, 2, ..., totalSeqLen - 1]` for the first pass
- For subsequent tokens: `initialSequenceLength + generatedTokenIds.length`
- Should position IDs account for the image/question boundary differently?

**Research Needed**:
- Verify how position IDs are calculated in the Hugging Face implementation
- Check if image embeddings and text embeddings have different position ID ranges

### 5. Attention Mask
**Question**: Is the attention mask correct for the concatenated sequence?

- **Current**: All ones `[1, 1, 1, ..., 1]` for the entire concatenated sequence
- Should image embeddings and question embeddings have different attention mask values?
- Should there be a boundary marker?

**Research Needed**:
- Check how attention masks are constructed in Hugging Face's implementation
- Verify if image and text embeddings need different attention mask handling

### 6. First Token Generation
**Question**: Why does the model generate garbage tokens from the start?

- The first generated token is already garbage (`"iosync"`, `"foundland"`)
- This suggests the issue is in the **first forward pass**, not the autoregressive loop
- Is the concatenated embedding incorrect? Is the prompt format wrong?

**Research Needed**:
- Compare the first token's logits distribution
- Verify the concatenated embedding shape and values
- Check if the model expects different input formatting

### 7. Model Compatibility
**Question**: Is the ONNX model compatible with our usage pattern?

- We're using `decoder_model_merged_int8.onnx` (INT8 quantized)
- The model might expect different input formats than what we're providing
- Are there any ONNX-specific requirements we're missing?

**Research Needed**:
- Check if there are ONNX-specific input/output requirements
- Verify if the quantized model has different behavior
- Check if there are example ONNX inference scripts for SmolVLM

### 8. Embedding Model Usage
**Question**: Are we using `embed_tokens.onnx` correctly?

- We use it to convert question token IDs to embeddings
- We use it to convert generated token IDs to embeddings for subsequent iterations
- Is this the correct approach, or should we use a different method?

**Research Needed**:
- Verify if `embed_tokens.onnx` is the official way to get embeddings
- Check if there are alternative methods (e.g., extracting embeddings from the decoder)
- Verify the embedding model's input/output format

## Code Locations

### Main Generation Function
- **File**: `src/models/smolvlm-256m.ts`
- **Function**: `generateResponse256M()` (line ~905)
- **Key Sections**:
  - Prompt formatting: `formatVQAPrompt256M()` (line ~875)
  - Image embedding concatenation: lines ~1238-1302
  - Autoregressive generation loop: lines ~1366-1936
  - Repetition detection: lines ~1570-1750

### Model Loading
- **File**: `src/models/smolvlm-256m.ts`
- **Function**: `loadSmolVLM256M()` (line ~471)
- Loads vision encoder, decoder, embedding model, and tokenizer

### Image Preprocessing
- **File**: `wasm-preprocess-256m/src/lib.rs`
- **Function**: `preprocess_image_for_smolvlm_256m()`
- Converts image to normalized Float32Array

## Logs Analysis

From the system logs, we can see:
- **Initial sequence length**: 75 (captioning) or 78 (VQA)
  - Image embeddings: ~64 tokens
  - Question tokens: ~11-14 tokens
  - Total: ~75-78 tokens
- **First token**: Already garbage (`"iosync"`, `"foundland"`)
- **Pattern emerges**: After ~10 tokens, the repetitive pattern starts
- **Repetition detection**: Not triggering early enough (generates 126+ tokens before stopping)

## Success Criteria

The model should:
1. Generate **coherent captions** for images (e.g., "A red car parked on a street")
2. Generate **relevant answers** to questions (e.g., "The image contains red, blue, and green colors")
3. **Stop naturally** at EOS tokens or when the answer is complete
4. **Not repeat** the same pattern indefinitely

## Next Steps

1. **Verify chat template**: Check the actual format from Hugging Face's tokenizer config
2. **Test embedding order**: Try `[question_embeds, image_embeds]` instead of `[image_embeds, question_embeds]`
3. **Check for image tokens**: Verify if the prompt needs image placeholder tokens
4. **Compare with working examples**: Find a working ONNX inference example for SmolVLM-256M
5. **Debug first token**: Investigate why the first generated token is already garbage
6. **Test with simpler prompts**: Try minimal prompts like just `"Assistant:"` or empty prompt

## Resources

- **Model Card**: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- **llama.cpp PR**: https://github.com/ggml-org/llama.cpp/pull/13050
- **ONNX Runtime Web**: https://onnxruntime.ai/docs/tutorials/web/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/en/model_doc/smolvlm

## Request for AI Assistants

Please help us:
1. **Identify the correct chat template format** for SmolVLM-256M-Instruct
2. **Determine the correct embedding concatenation order** (image first vs. question first)
3. **Verify if image tokens are needed** in the prompt
4. **Debug why the first token is garbage** (suggests first forward pass issue)
5. **Find working examples** of SmolVLM-256M ONNX inference
6. **Suggest alternative approaches** if our current method is fundamentally wrong

Any insights, code examples, or research findings would be greatly appreciated!

