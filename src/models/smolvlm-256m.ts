// SmolVLM-256M integration using ONNX Runtime Web
// Handles model loading, tokenization, chat template, and inference

import * as ort from 'onnxruntime-web';

// SmolVLM-256M model configuration
// ONNX models are in the 'onnx' subdirectory
const SMOLVLM_MODEL_BASE_URL = 'https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx';
const TARGET_IMAGE_SIZE = 512; // SmolVLM-256M uses 512x512 for vision encoder

/**
 * Tensor shape types for type safety
 * These types ensure correct tensor dimensions and prevent rank mismatches
 */

/**
 * Past key-value cache shape: [batch, num_heads, seq_len, head_dim]
 * For first forward pass, seq_len = 0
 */
type PastKeyValueShape = [number, number, number, number];

/**
 * Image tensor shape: [batch, num_images, channels, height, width]
 */
type ImageTensorShape = [number, number, number, number, number];

/**
 * Pixel attention mask shape: [batch, num_images, height, width]
 */
type PixelAttentionMaskShape = [number, number, number, number];

/**
 * Decoder attention mask shape: [batch, sequence_length]
 */
type DecoderAttentionMaskShape = [number, number];

/**
 * Position IDs shape: [batch, sequence_length]
 */
type PositionIdsShape = [number, number];

/**
 * Input IDs tensor shape: [batch, sequence_length]
 */
type InputIdsShape = [number, number];

// CORS proxy services for Hugging Face model loading
// These proxies allow cross-origin requests to Hugging Face CDN
// Format: { baseUrl: string, returnsJson: boolean }
// returnsJson: true if proxy returns JSON-wrapped content (e.g., { contents: "..." })
const CORS_PROXY_SERVICES = [
  { baseUrl: 'https://api.allorigins.win/raw?url=', returnsJson: false },
  { baseUrl: 'https://corsproxy.io/?', returnsJson: false },
  { baseUrl: 'https://api.codetabs.com/v1/proxy?quest=', returnsJson: false },
  { baseUrl: 'https://cors-anywhere.herokuapp.com/', returnsJson: false },
  // Whatever Origin moved to end - may not work well for very large binary files
  { baseUrl: 'https://whateverorigin.org/get?url=', returnsJson: true },
] as const;

/**
 * Check if a URL needs CORS proxying
 */
function needsProxy(url: string): boolean {
  return (
    url.includes('huggingface.co') &&
    !url.includes('cdn.jsdelivr.net') &&
    !url.includes('api.allorigins.win') &&
    !url.includes('corsproxy.io') &&
    !url.includes('whateverorigin.org') &&
    !url.includes('api.codetabs.com') &&
    !url.includes('cors-anywhere.herokuapp.com')
  );
}

interface SmolVLMModel {
  visionEncoder: ort.InferenceSession | null;
  decoder: ort.InferenceSession | null;
  embedTokens: ort.InferenceSession | null;
  tokenizer: {
    encode: (text: string) => { ids: number[] };
    decode: (ids: number[]) => string;
  } | null;
  isLoaded: boolean;
}

const smolvlmModel: SmolVLMModel = {
  visionEncoder: null,
  decoder: null,
  embedTokens: null,
  tokenizer: null,
  isLoaded: false,
};

/**
 * Progress callback type for model loading
 */
export type ProgressCallback = (progress: number, message: string) => void;

/**
 * Logging callback type for download operations
 */
export type LogCallback = (message: string, type: 'info' | 'success' | 'warning' | 'error') => void;

/**
 * Cache name for storing model files
 */
const CACHE_NAME = 'smolvlm-models-v1';

/**
 * Check if a URL exists in cache
 */
async function checkCache(url: string): Promise<Response | null> {
  try {
    if ('caches' in window) {
      const cache = await caches.open(CACHE_NAME);
      const cachedResponse = await cache.match(url);
      if (cachedResponse && cachedResponse.ok) {
        return cachedResponse;
      }
    }
  } catch {
    // Cache API not available or error accessing cache
    // Silently fail and proceed with download
  }
  return null;
}

/**
 * Save a response to cache
 */
async function saveToCache(url: string, response: Response): Promise<void> {
  try {
    if ('caches' in window) {
      const cache = await caches.open(CACHE_NAME);
      // Clone the response because it can only be consumed once
      await cache.put(url, response.clone());
    }
  } catch {
    // Cache API not available or error saving to cache
    // Silently fail - caching is optional
  }
}

/**
 * Helper function to fetch with progress tracking, CORS proxy support, and caching
 */
async function fetchWithProgress(
  url: string,
  onProgress: (loaded: number, total: number) => void,
  onLog?: LogCallback,
  isTextResponse?: boolean
): Promise<Response> {
  // Check cache first
  const cachedResponse = await checkCache(url);
  if (cachedResponse) {
    if (onLog) {
      onLog(`Found in cache: ${url}`, 'success');
    }
    // Return cached response - no progress tracking needed for cached files
    return cachedResponse;
  }
  
  // Determine if we need to use a CORS proxy
  const useProxy = needsProxy(url);
  
  let response: Response | undefined;
  
  if (useProxy) {
    // Try each proxy service in order
    let lastError: Error | null = null;
    
    if (onLog) {
      onLog(`Initiating download via CORS proxy. Target: ${url}`, 'info');
    }
    
    for (const proxy of CORS_PROXY_SERVICES) {
      try {
        const proxyUrl = proxy.baseUrl + encodeURIComponent(url);
        if (onLog) {
          onLog(`Attempting proxy: ${proxy.baseUrl} → ${url}`, 'info');
        }
        const proxyResponse = await fetch(proxyUrl, {
          redirect: 'follow'
        });
        
        // Skip proxies that return error status codes (408 timeout, 500 server error, etc.)
        // Also skip redirects (301, 302) as they indicate the proxy isn't working correctly
        if (proxyResponse.status === 301 || proxyResponse.status === 302 || proxyResponse.status === 303 || proxyResponse.status === 307 || proxyResponse.status === 308 || proxyResponse.status === 408 || proxyResponse.status === 403 || proxyResponse.status === 500 || proxyResponse.status === 502 || proxyResponse.status === 503 || proxyResponse.status === 504) {
          lastError = new Error(`Proxy ${proxy.baseUrl} returned error status ${proxyResponse.status}: ${proxyResponse.statusText}`);
          if (onLog) {
            const statusType = (proxyResponse.status >= 301 && proxyResponse.status <= 308) ? 'redirect' : 'error';
            onLog(`Proxy failed: ${proxy.baseUrl} (Status: ${proxyResponse.status} ${proxyResponse.statusText} - ${statusType})`, 'warning');
          }
          continue;
        }
        
        // Check if proxy response is OK
        if (proxyResponse.ok || proxyResponse.status === 200) {
          // Handle JSON-wrapped responses (like Whatever Origin)
          if (proxy.returnsJson) {
            const jsonData: unknown = await proxyResponse.json();
            // Type guard: check if response has expected structure
            if (jsonData && typeof jsonData === 'object' && jsonData !== null && 'contents' in jsonData) {
              const contentsProperty = Object.getOwnPropertyDescriptor(jsonData, 'contents');
              if (!contentsProperty || typeof contentsProperty.value !== 'string') {
                lastError = new Error(`Proxy ${proxy.baseUrl} returned JSON with non-string contents`);
                continue;
              }
              const contents: string = contentsProperty.value;
              // Create a new Response from the unwrapped content
              // Whatever Origin returns content as a string
              // For binary data, we need to convert the string to bytes
              // If it's base64 encoded, decode it; otherwise treat as raw bytes
              let binaryBuffer: ArrayBuffer;
              try {
                // Try to decode as base64 first (common for binary data in JSON)
                const decoded = atob(contents);
                binaryBuffer = new ArrayBuffer(decoded.length);
                const view = new Uint8Array(binaryBuffer);
                for (let i = 0; i < decoded.length; i++) {
                  view[i] = decoded.charCodeAt(i);
                }
              } catch {
                // If base64 decode fails, treat as raw string (UTF-8 encoded bytes)
                const encoder = new TextEncoder();
                const encoded = encoder.encode(contents);
                // Create a new ArrayBuffer and copy data
                binaryBuffer = new ArrayBuffer(encoded.length);
                const view = new Uint8Array(binaryBuffer);
                view.set(encoded);
              }
              const blob = new Blob([binaryBuffer], { type: 'application/octet-stream' });
              response = new Response(blob, {
                status: 200,
                statusText: 'OK',
                headers: proxyResponse.headers,
              });
              break;
            } else {
              // JSON response doesn't have expected structure
              lastError = new Error(`Proxy ${proxy.baseUrl} returned unexpected JSON structure`);
              continue;
            }
          } else {
            // For text/JSON responses, skip binary validation
            // For binary responses, validate that proxy returned binary data, not HTML error page
            if (isTextResponse) {
              // Text responses can be any size, just check for HTML
              const clonedResponse = proxyResponse.clone();
              const firstChunk = await clonedResponse.arrayBuffer();
              const firstBytes = new Uint8Array(firstChunk).slice(0, Math.min(10, firstChunk.byteLength));
              
              // Check if response looks like HTML (starts with '<' or '<!')
              if (firstBytes.length > 0 && (firstBytes[0] === 0x3C || (firstBytes[0] === 0x3C && firstBytes.length > 1 && firstBytes[1] === 0x21))) {
                // This looks like HTML, not text data - try next proxy
                const textDecoder = new TextDecoder();
                const preview = textDecoder.decode(new Uint8Array(firstChunk).slice(0, 200));
                lastError = new Error(
                  `Proxy returned HTML instead of text data. Content preview: ${preview.substring(0, 100)}`
                );
                if (onLog) {
                  onLog(`Proxy returned HTML instead of text data`, 'warning');
                }
                continue;
              }
              
              // Response looks valid, use the original (unconsumed) response
              if (onLog) {
                onLog(`Proxy connection established: ${proxy.baseUrl} → ${url}`, 'success');
              }
              response = proxyResponse;
              break;
            } else {
              // Validate that proxy returned binary data, not HTML error page
              // Use clone to peek without consuming original stream
              const clonedResponse = proxyResponse.clone();
              const firstChunk = await clonedResponse.arrayBuffer();
              
              // Check file size first - ONNX models are large (several MB)
              if (firstChunk.byteLength < 1000) {
                // File is suspiciously small - likely an error message
                const textDecoder = new TextDecoder();
                const preview = textDecoder.decode(new Uint8Array(firstChunk).slice(0, 200));
                lastError = new Error(
                  `Proxy returned suspiciously small file (${firstChunk.byteLength} bytes). ` +
                  `Expected ONNX model to be much larger. Content: ${preview.substring(0, 100)}`
                );
                if (onLog) {
                  onLog(`Proxy returned suspiciously small file (${firstChunk.byteLength} bytes)`, 'warning');
                }
                continue;
              }
              
              const firstBytes = new Uint8Array(firstChunk).slice(0, Math.min(10, firstChunk.byteLength));
              
              // Check if response looks like HTML (starts with '<' or '<!')
              if (firstBytes.length > 0 && (firstBytes[0] === 0x3C || (firstBytes[0] === 0x3C && firstBytes.length > 1 && firstBytes[1] === 0x21))) {
                // This looks like HTML, not binary data - try next proxy
                const textDecoder = new TextDecoder();
                const preview = textDecoder.decode(new Uint8Array(firstChunk).slice(0, 200));
                lastError = new Error(
                  `Proxy returned HTML instead of binary data. Content preview: ${preview.substring(0, 100)}`
                );
                if (onLog) {
                  onLog(`Proxy returned HTML instead of binary data`, 'warning');
                }
                continue;
              }
              
              // Response looks valid, use the original (unconsumed) response
              if (onLog) {
                onLog(`Proxy connection established: ${proxy.baseUrl} → ${url}`, 'success');
              }
              response = proxyResponse;
              break;
            }
          }
        }
        // If response is not OK, try next proxy
        lastError = new Error(`Proxy returned status ${proxyResponse.status}: ${proxyResponse.statusText}`);
      } catch (proxyError) {
        // Try next proxy on error
        lastError = proxyError instanceof Error ? proxyError : new Error('Unknown proxy error');
        if (onLog) {
          const errorMsg = proxyError instanceof Error ? proxyError.message : 'Unknown error';
          onLog(`Proxy error: ${errorMsg}`, 'warning');
        }
        continue;
      }
    }
    
    // If all proxies failed, try direct fetch as last resort
    if (!response) {
      if (onLog) {
        onLog('All CORS proxies failed. Attempting direct fetch...', 'warning');
      }
      try {
        response = await fetch(url);
      } catch (directError) {
        const directErrorMsg = directError instanceof Error ? directError.message : 'Unknown error';
        const proxyErrorMsg = lastError ? lastError.message : 'All proxies failed';
        if (onLog) {
          onLog(`All CORS proxies failed (${CORS_PROXY_SERVICES.length} tried)`, 'error');
          onLog(`Direct fetch also failed: ${directErrorMsg}`, 'error');
        }
        throw new Error(
          `Failed to fetch ${url} from Hugging Face. ` +
          `All CORS proxies failed (${CORS_PROXY_SERVICES.length} tried). ` +
          `Proxy errors: ${proxyErrorMsg}. ` +
          `Direct fetch also failed: ${directErrorMsg}. ` +
          `This is likely due to CORS restrictions or network issues. ` +
          `Try: refreshing the page, checking your network connection, or using a different browser.`
        );
      }
    }
  } else {
    // Direct fetch for non-Hugging Face URLs or already-proxied URLs
    response = await fetch(url);
  }
  
  // At this point, response should be assigned
  if (!response) {
    throw new Error('Failed to obtain response from any source');
  }
  
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? Number.parseInt(contentLength, 10) : 0;

  if (!response.body) {
    throw new Error('Response body is null');
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastChunkTime = Date.now();
  const TIMEOUT_MS = 30000; // 30 seconds timeout
  const PROGRESS_INTERVAL_MS = 2000; // Report progress every 2 seconds for unknown sizes
  let lastProgressReportTime = Date.now();
  let lastProgressReportBytes = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    
    // Check for timeout (no data received for 30 seconds)
    const now = Date.now();
    if (value) {
      lastChunkTime = now;
    } else if (now - lastChunkTime > TIMEOUT_MS) {
      throw new Error(
        `Download stalled: No data received for ${Math.floor((now - lastChunkTime) / 1000)} seconds. ` +
        `Loaded ${Math.floor(loaded / 1024 / 1024)} MB so far. ` +
        `This may indicate a network issue or server problem. Try refreshing the page.`
      );
    }
    
    if (done) {
      break;
    }
    if (value) {
      chunks.push(value);
      loaded += value.length;
      
      // Report progress: always when total is known, periodically when unknown
      if (total > 0) {
        onProgress(loaded, total);
      } else {
        // Report progress periodically for unknown sizes (every 2 seconds or every 1MB)
        const timeSinceLastReport = now - lastProgressReportTime;
        const bytesSinceLastReport = loaded - lastProgressReportBytes;
        const shouldReport = (timeSinceLastReport >= PROGRESS_INTERVAL_MS) || 
                            (bytesSinceLastReport >= 1024 * 1024);
        if (shouldReport) {
          onProgress(loaded, 0);
          lastProgressReportTime = now;
          lastProgressReportBytes = loaded;
        }
      }
    }
  }

  // Reconstruct the response with the accumulated chunks
  const allChunks = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    allChunks.set(chunk, offset);
    offset += chunk.length;
  }

  // Validate that we received the expected data type (not HTML error page)
  if (allChunks.length > 0) {
    const firstBytes = allChunks.slice(0, Math.min(20, allChunks.length));
    
    // Check if response looks like HTML (starts with '<' or '<!')
    if (firstBytes[0] === 0x3C || (firstBytes[0] === 0x3C && firstBytes.length > 1 && firstBytes[1] === 0x21)) {
      const textDecoder = new TextDecoder();
      const preview = textDecoder.decode(allChunks.slice(0, Math.min(500, allChunks.length)));
      throw new Error(
        `Received HTML instead of expected data. ` +
        `This usually means the CORS proxy returned an error page. ` +
        `Response preview: ${preview.substring(0, 200)}... ` +
        `Try refreshing the page or check your network connection.`
      );
    }
    
    if (isTextResponse) {
      // For text/JSON responses, validate it looks like JSON (starts with '{' or '[')
      const textDecoder = new TextDecoder();
      const textStart = textDecoder.decode(firstBytes).trim();
      if (textStart.length > 0 && textStart[0] !== '{' && textStart[0] !== '[') {
        const preview = textDecoder.decode(allChunks.slice(0, Math.min(500, allChunks.length)));
        throw new Error(
          `Received unexpected text format. Expected JSON (starting with '{' or '['), ` +
          `but got: ${preview.substring(0, 200)}... ` +
          `This may indicate a proxy error or corrupted download.`
        );
      }
      // JSON files can be small, so skip size validation for text responses
    } else {
      // For binary ONNX files, validate size and format
      // ONNX files are protobuf-encoded and typically start with specific patterns
      // If the file is suspiciously small, warn
      if (allChunks.length < 100) {
        throw new Error(
          `Received suspiciously small file (${allChunks.length} bytes). ` +
          `ONNX model files should be much larger. This suggests the download was incomplete or corrupted.`
        );
      }
    }
  }

  // Create new response with proper content-type for binary data
  const newHeaders = new Headers(response.headers);
  // Ensure content-type is set correctly for binary data
  if (!newHeaders.has('content-type')) {
    newHeaders.set('content-type', 'application/octet-stream');
  }

  const finalResponse = new Response(allChunks, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders,
  });
  
  // Save to cache for future use
  await saveToCache(url, finalResponse.clone());
  
  return finalResponse;
}

/**
 * Load SmolVLM-256M ONNX models and tokenizer with progress tracking
 * @param onProgress - Optional callback to report loading progress (0-100)
 */
export async function loadSmolVLM256M(onProgress?: ProgressCallback, onLog?: LogCallback): Promise<void> {
  if (smolvlmModel.isLoaded) {
    return;
  }

  const reportProgress = (progress: number, message: string): void => {
    if (onProgress) {
      onProgress(progress, message);
    }
  };

  try {
    // Initialize ONNX Runtime - show starting indicator
    reportProgress(0, 'Initializing ONNX Runtime...');
    
    // Configure ONNX Runtime Web WASM paths - MUST be set before first session creation
    // In dev mode, use node_modules path (Vite serves it correctly)
    // In production, use public directory path (files are copied to dist/)
    // Use path prefix approach: ONNX Runtime will append the correct filename
    // The default build uses 'ort-wasm-simd-threaded.wasm' and 'ort-wasm-simd-threaded.mjs'
    // Note: Path must end with '/' so ONNX Runtime can properly concatenate the filename
    // Detect environment: in dev, Vite serves node_modules; in production, use public directory
    // Check if we're in dev by looking for Vite's dev server indicator
    const isDev = typeof window !== 'undefined' && window.location.hostname === 'localhost';
    const wasmBasePath = isDev 
      ? '/node_modules/onnxruntime-web/dist/' 
      : '/onnxruntime-wasm/';
    
    // Set wasmPaths as a path prefix string
    // ONNX Runtime Web will automatically append the correct filename based on build type
    ort.env.wasm.wasmPaths = wasmBasePath;
    
    // Configure WASM performance settings
    // Use SIMD for better performance if available
    ort.env.wasm.simd = true;
    // Use single thread for now (can be increased if needed)
    ort.env.wasm.numThreads = 1;
    
    reportProgress(1, 'ONNX Runtime WASM configured');
    
    // Configure execution providers: WASM first (most reliable), then WebGPU if available
    const executionProviders: string[] = [];
    
    // Always include WASM as primary backend (most reliable)
    executionProviders.push('wasm');
    
    // Check for WebGPU support and add as secondary option
    if ('gpu' in navigator) {
      executionProviders.push('webgpu');
    }
    
    // Report that ONNX Runtime is ready to start loading models
    reportProgress(2, 'ONNX Runtime ready');

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders,
      graphOptimizationLevel: 'all',
    };

    // Load vision encoder model (estimated 30% of total, starting from 5%)
    reportProgress(5, 'Loading vision encoder model...');
    if (onLog) {
      onLog('Checking cache for vision encoder...', 'info');
    }
    const visionEncoderUrl = `${SMOLVLM_MODEL_BASE_URL}/vision_encoder.onnx`;
    
    // Check cache first
    const cachedVisionEncoder = await checkCache(visionEncoderUrl);
    let visionEncoderFromCache = false;
    if (cachedVisionEncoder) {
      if (onLog) {
        onLog('Vision encoder found in cache', 'success');
      }
      visionEncoderFromCache = true;
      reportProgress(35, 'Vision encoder loaded from cache ✓');
    } else {
      if (onLog) {
        onLog('Starting vision encoder download...', 'info');
      }
    }
    
    // Fetch model with progress tracking (or use cached version)
    // Progress range: 5% to 35% (30% of total for vision encoder)
    let visionEncoderProgress = 5;
    const visionEncoderResponse = cachedVisionEncoder || await fetchWithProgress(
      visionEncoderUrl,
      (loaded, total) => {
        if (total > 0) {
          // Known size: show percentage
          visionEncoderProgress = 5 + Math.floor((loaded / total) * 30);
          const loadedMB = Math.floor(loaded / 1024 / 1024);
          const totalMB = Math.floor(total / 1024 / 1024);
          reportProgress(visionEncoderProgress, `Loading vision encoder... ${visionEncoderProgress}% (${loadedMB}/${totalMB} MB)`);
        } else {
          // Unknown size: show bytes loaded
          const loadedMB = Math.floor(loaded / 1024 / 1024);
          // Estimate progress based on typical vision encoder size (~393MB)
          const estimatedTotal = 400 * 1024 * 1024; // 400MB estimate
          const estimatedProgress = Math.min(35, 5 + Math.floor((loaded / estimatedTotal) * 30));
          reportProgress(estimatedProgress, `Loading vision encoder... ${loadedMB} MB`);
        }
      },
      onLog
    );

    const visionEncoderArrayBuffer = await visionEncoderResponse.arrayBuffer();
    if (onLog) {
      const sizeMB = Math.floor(visionEncoderArrayBuffer.byteLength / 1024 / 1024);
      if (visionEncoderFromCache) {
        onLog(`Vision encoder loaded from cache: ${sizeMB} MB`, 'success');
      } else {
        onLog(`Vision encoder downloaded: ${sizeMB} MB`, 'success');
      }
    }
    reportProgress(35, 'Initializing vision encoder...');
    if (onLog) {
      onLog('Initializing vision encoder session...', 'info');
    }
    smolvlmModel.visionEncoder = await ort.InferenceSession.create(visionEncoderArrayBuffer, sessionOptions);
    reportProgress(35, 'Vision encoder loaded ✓');
    if (onLog) {
      onLog('Vision encoder initialized successfully', 'success');
    }

    // Load decoder model (estimated 60% of total, from 35% to 95%)
    // Using quantized INT8 decoder for smaller file size (~4x smaller than FP32)
    reportProgress(35, 'Loading decoder model (quantized INT8)...');
    if (onLog) {
      onLog('Checking cache for decoder model...', 'info');
    }
    const decoderUrl = `${SMOLVLM_MODEL_BASE_URL}/decoder_model_merged_int8.onnx`;
    
    // Check cache first
    const cachedDecoder = await checkCache(decoderUrl);
    let decoderFromCache = false;
    if (cachedDecoder) {
      if (onLog) {
        onLog('Decoder model found in cache', 'success');
      }
      decoderFromCache = true;
      reportProgress(95, 'Decoder loaded from cache ✓');
    } else {
      if (onLog) {
        onLog('Starting decoder model download (INT8 quantized)...', 'info');
      }
    }
    
    let decoderProgress = 35;
    const decoderResponse = cachedDecoder || await fetchWithProgress(
      decoderUrl,
      (loaded, total) => {
        if (total > 0) {
          // Known size: show percentage
          decoderProgress = 35 + Math.floor((loaded / total) * 60);
          const loadedMB = Math.floor(loaded / 1024 / 1024);
          const totalMB = Math.floor(total / 1024 / 1024);
          reportProgress(decoderProgress, `Loading decoder... ${decoderProgress}% (${loadedMB}/${totalMB} MB)`);
        } else {
          // Unknown size: show bytes loaded
          const loadedMB = Math.floor(loaded / 1024 / 1024);
          // Estimate progress based on typical quantized INT8 decoder size (~350MB, ~4x smaller than FP32)
          // Use a conservative estimate to avoid showing 100% prematurely
          const estimatedTotal = 400 * 1024 * 1024; // 400MB estimate for INT8 quantized
          const estimatedProgress = Math.min(95, 35 + Math.floor((loaded / estimatedTotal) * 60));
          reportProgress(estimatedProgress, `Loading decoder (INT8)... ${loadedMB} MB`);
        }
      },
      onLog
    );

    const decoderArrayBuffer = await decoderResponse.arrayBuffer();
    if (onLog) {
      const sizeMB = Math.floor(decoderArrayBuffer.byteLength / 1024 / 1024);
      if (decoderFromCache) {
        onLog(`Decoder model loaded from cache: ${sizeMB} MB`, 'success');
      } else {
        onLog(`Decoder model downloaded: ${sizeMB} MB`, 'success');
      }
    }
    
    // Validate ArrayBuffer size (ONNX files should be reasonably large - typically several MB)
    if (decoderArrayBuffer.byteLength < 100000) {
      throw new Error(
        `Decoder model file is too small (${decoderArrayBuffer.byteLength} bytes). ` +
        `Expected at least 100KB. This suggests the file was not downloaded correctly or is corrupted. ` +
        `The file may be an HTML error page instead of binary data.`
      );
    }
    
    // Additional validation: check first bytes to ensure it's binary data
    const firstBytes = new Uint8Array(decoderArrayBuffer).slice(0, 10);
    if (firstBytes[0] === 0x3C) {
      const textDecoder = new TextDecoder();
      const preview = textDecoder.decode(new Uint8Array(decoderArrayBuffer).slice(0, 200));
      throw new Error(
        `Decoder model appears to be HTML/text instead of binary data. ` +
        `First 200 chars: ${preview.substring(0, 100)}... ` +
        `This suggests the CORS proxy returned an error page.`
      );
    }
    
    reportProgress(95, 'Initializing decoder...');
    if (onLog) {
      onLog('Initializing decoder session...', 'info');
    }
    try {
      smolvlmModel.decoder = await ort.InferenceSession.create(decoderArrayBuffer, sessionOptions);
    } catch (sessionError) {
      const errorMsg = sessionError instanceof Error ? sessionError.message : 'Unknown error';
      throw new Error(
        `Failed to create decoder session: ${errorMsg}. ` +
        `The model file may be corrupted or in an unsupported format. ` +
        `File size: ${decoderArrayBuffer.byteLength} bytes.`
      );
    }
    reportProgress(95, 'Decoder loaded ✓');
    if (onLog) {
      onLog('Decoder initialized successfully', 'success');
    }

    // Try to load embed_tokens.onnx if it exists (for converting token IDs to embeddings)
    // This allows us to properly concatenate image embeddings with question embeddings
    try {
      const embedTokensUrl = `${SMOLVLM_MODEL_BASE_URL}/embed_tokens.onnx`;
      if (onLog) {
        onLog('Checking for embedding model...', 'info');
      }
      const cachedEmbedTokens = await checkCache(embedTokensUrl);
      const embedTokensResponse = cachedEmbedTokens || await fetchWithProgress(
        embedTokensUrl,
        () => {
          // Silent progress - this is optional
        },
        onLog
      );
      
      if (embedTokensResponse.ok) {
        const embedTokensArrayBuffer = await embedTokensResponse.arrayBuffer();
        if (embedTokensArrayBuffer.byteLength > 1000) {
          // Valid file size
          smolvlmModel.embedTokens = await ort.InferenceSession.create(embedTokensArrayBuffer, sessionOptions);
          if (onLog) {
            const sizeMB = Math.floor(embedTokensArrayBuffer.byteLength / 1024 / 1024);
            const sizeKB = Math.floor(embedTokensArrayBuffer.byteLength / 1024);
            if (sizeMB > 0) {
              onLog(`Embedding model loaded successfully: ${sizeMB} MB`, 'success');
            } else {
              onLog(`Embedding model loaded successfully: ${sizeKB} KB`, 'success');
            }
          }
        } else {
          if (onLog) {
            onLog(`Embedding model file too small (${embedTokensArrayBuffer.byteLength} bytes), skipping`, 'warning');
          }
        }
      } else {
        if (onLog) {
          onLog(`Embedding model not found (${embedTokensResponse.status}), will use input_ids if supported`, 'warning');
        }
      }
    } catch (embedError) {
      // Embedding model is optional - if it doesn't exist, we'll use input_ids approach
      const errorMsg = embedError instanceof Error ? embedError.message : 'Unknown error';
      if (onLog) {
        onLog(`Embedding model not available (${errorMsg}), will use input_ids if supported`, 'warning');
      }
    }

    // Load tokenizer (estimated 5% of total, from 95% to 100%)
    // Tokenizer is in the root directory, not in the onnx subdirectory
    reportProgress(95, 'Loading tokenizer...');
    if (onLog) {
      onLog('Checking cache for tokenizer...', 'info');
    }
    const tokenizerUrl = 'https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/tokenizer.json';
    
    // Check cache first
    const cachedTokenizer = await checkCache(tokenizerUrl);
    let tokenizerFromCache = false;
    if (cachedTokenizer) {
      if (onLog) {
        onLog('Tokenizer found in cache', 'success');
      }
      tokenizerFromCache = true;
      reportProgress(100, 'Tokenizer loaded from cache ✓');
    } else {
      if (onLog) {
        onLog('Starting tokenizer download...', 'info');
      }
    }
    
    let tokenizerProgress = 95;
    const tokenizerResponse = cachedTokenizer || await fetchWithProgress(
      tokenizerUrl,
      (loaded, total) => {
        if (total > 0) {
          // Known size: show percentage
          tokenizerProgress = 95 + Math.floor((loaded / total) * 5);
          const loadedKB = Math.floor(loaded / 1024);
          const totalKB = Math.floor(total / 1024);
          reportProgress(tokenizerProgress, `Loading tokenizer... ${tokenizerProgress}% (${loadedKB}/${totalKB} KB)`);
        } else {
          // Unknown size: show bytes loaded
          const loadedKB = Math.floor(loaded / 1024);
          // Estimate progress based on typical tokenizer size (~3.5MB)
          const estimatedTotal = 4 * 1024 * 1024; // 4MB estimate
          const estimatedProgress = Math.min(100, 95 + Math.floor((loaded / estimatedTotal) * 5));
          reportProgress(estimatedProgress, `Loading tokenizer... ${loadedKB} KB`);
        }
      },
      onLog,
      true  // Indicate this is a text/JSON response
    );

    if (!tokenizerResponse.ok) {
      throw new Error(`Failed to load tokenizer: ${tokenizerResponse.statusText}`);
    }
    
    // Read as ArrayBuffer first to get accurate download size
    const tokenizerArrayBuffer = await tokenizerResponse.arrayBuffer();
    const sizeMB = Math.floor(tokenizerArrayBuffer.byteLength / 1024 / 1024);
    const sizeKB = Math.floor(tokenizerArrayBuffer.byteLength / 1024);
    
    // Log the size
    if (onLog) {
      if (sizeMB >= 1) {
        if (tokenizerFromCache) {
          onLog(`Tokenizer loaded from cache: ${sizeMB} MB`, 'success');
        } else {
          onLog(`Tokenizer downloaded: ${sizeMB} MB`, 'success');
        }
      } else {
        if (tokenizerFromCache) {
          onLog(`Tokenizer loaded from cache: ${sizeKB} KB`, 'success');
        } else {
          onLog(`Tokenizer downloaded: ${sizeKB} KB`, 'success');
        }
      }
    }
    
    // Parse JSON from the ArrayBuffer
    const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(tokenizerArrayBuffer));
    const { Tokenizer } = await import('@huggingface/tokenizers');
    
    // Tokenizer constructor takes (tokenizer: Object, config: Object)
    // The tokenizer.json from Hugging Face contains both the tokenizer and config
    // Type guard: ensure tokenizerJson is an object
    if (typeof tokenizerJson !== 'object' || tokenizerJson === null) {
      throw new Error('Tokenizer JSON is not a valid object');
    }
    
    // Type guard: ensure tokenizerJson is a Record
    if (typeof tokenizerJson !== 'object' || tokenizerJson === null) {
      throw new Error('Tokenizer JSON is not a valid object');
    }
    
    // Use Object constructor to satisfy Tokenizer constructor requirements
    // We've already verified tokenizerJson is an object, so we can safely iterate
    const tokenizerConfig: Record<string, unknown> = {};
    // Create a properly typed record by iterating over the object
    const keys = Object.keys(tokenizerJson);
    for (const key of keys) {
      const descriptor = Object.getOwnPropertyDescriptor(tokenizerJson, key);
      if (descriptor && descriptor.value !== undefined) {
        tokenizerConfig[key] = descriptor.value;
      }
    }
    
    const tokenizer = new Tokenizer(tokenizerConfig, tokenizerConfig);
    
    // Wrap tokenizer with encode/decode methods
    if (onLog) {
      onLog('Tokenizer downloaded and parsed', 'success');
    }
    smolvlmModel.tokenizer = {
      encode: (text: string) => {
        const encoded = tokenizer.encode(text);
        // Encoding has ids property (Uint32Array or number[])
        const ids = encoded.ids;
        return { ids: Array.isArray(ids) ? ids : Array.from(ids) };
      },
      decode: (ids: number[]) => {
        return tokenizer.decode(ids);
      },
    };

    reportProgress(100, 'Tokenizer loaded ✓');
    smolvlmModel.isLoaded = true;
    if (onLog) {
      onLog('SmolVLM-256M model fully loaded and ready', 'success');
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    if (onLog) {
      onLog(`Model loading failed: ${errorMsg}`, 'error');
    }
    throw new Error(`Failed to load SmolVLM-256M: ${errorMsg}`);
  }
}

/**
 * Format VQA prompt using SmolVLM chat template
 * SmolVLM-256M-Instruct uses a specific format with image and text content
 */
export function formatVQAPrompt256M(question: string): string {
  // SmolVLM chat template format based on official Hugging Face implementation
  // Format: "<|im_start|>User: <image> {question}<end_of_utterance>\nAssistant:"
  // The <image> token will be replaced by image embeddings during the merge step
  return `<|im_start|>User: <image> ${question}<end_of_utterance>\nAssistant:`;
}

/**
 * Tokenize text using the model's tokenizer
 */
export function tokenizeText256M(text: string): number[] {
  if (!smolvlmModel.tokenizer) {
    throw new Error('Tokenizer not loaded');
  }

  const encoded = smolvlmModel.tokenizer.encode(text);
  return encoded.ids;
}

/**
 * Generate response using SmolVLM-256M
 * @param imageData - Preprocessed image data from WASM (normalized Float32Array)
 * @param question - User's question about the image
 * @param onLog - Optional callback for logging messages to System Logs
 * @returns Generated answer text
 */
export async function generateResponse256M(
  imageData: Float32Array,
  question: string,
  onLog?: LogCallback
): Promise<string> {
  if (!smolvlmModel.isLoaded || !smolvlmModel.visionEncoder || !smolvlmModel.decoder) {
    throw new Error('SmolVLM-256M model not loaded');
  }

  if (!smolvlmModel.tokenizer) {
    throw new Error('Tokenizer not loaded');
  }

  // Format and tokenize the question
  // Based on official Hugging Face example, for captioning use "Can you describe this image?"
  // The processor.apply_chat_template() formats this as: "<|user|>\nCan you describe this image?\n<|assistant|>\n"
  const prompt = question.length > 0 
    ? formatVQAPrompt256M(question)
    : formatVQAPrompt256M('Can you describe this image?');
  const questionTokenIds = tokenizeText256M(prompt);
  
  // Find the <image> token ID and its index in the tokenized sequence
  // This is critical for the conditional merge: we need to replace this token's embedding
  let imageTokenId: number;
  try {
    const imageTokenEncoded = smolvlmModel.tokenizer.encode('<image>');
    if (imageTokenEncoded.ids.length === 0) {
      throw new Error('Could not encode <image> token');
    }
    imageTokenId = imageTokenEncoded.ids[0];
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Failed to encode <image> token: ${errorMsg}`);
  }
  
  const imageTokenIndex = questionTokenIds.indexOf(imageTokenId);
  if (imageTokenIndex === -1) {
    throw new Error(`Could not find <image> token (ID: ${imageTokenId}) in tokenized prompt. Token IDs: [${questionTokenIds.join(', ')}]`);
  }
  
  if (onLog) {
    onLog(`Starting generation. Prompt: "${prompt}"`, 'info');
    onLog(`Question token IDs (${questionTokenIds.length} tokens): [${questionTokenIds.slice(0, 10).join(', ')}${questionTokenIds.length > 10 ? '...' : ''}]`, 'info');
    onLog(`<image> token ID: ${imageTokenId}, found at index: ${imageTokenIndex}`, 'info');
  }

  // Prepare image tensor for vision encoder
  // Shape: [1, 1, 3, height, width] (batch, num_images, channels, height, width)
  // Note: imageData from WASM is [R, G, B, R, G, B, ...] flattened
  // Need to reshape to [1, 1, 3, height, width] format (5D tensor)
  const imageHeight = TARGET_IMAGE_SIZE;
  const imageWidth = TARGET_IMAGE_SIZE;
  
  // Reshape image data from [H*W*3] to [1, 1, 3, H, W]
  // ONNX expects channels-first format with batch and num_images dimensions
  // Index calculation for [batch, num_images, channel, height, width]:
  const IMAGE_BATCH_SIZE = 1;
  const IMAGE_NUM_IMAGES = 1;
  const NUM_CHANNELS = 3;
  const CHANNEL_RED = 0;
  const CHANNEL_GREEN = 1;
  const CHANNEL_BLUE = 2;
  const IMAGE_TENSOR_DATA_TYPE = 'float32';
  const BATCH_INDEX = 0;
  const NUM_IMAGES_INDEX = 0;
  
  const reshapedData = new Float32Array(IMAGE_BATCH_SIZE * IMAGE_NUM_IMAGES * NUM_CHANNELS * imageHeight * imageWidth);
  const batchStride = IMAGE_NUM_IMAGES * NUM_CHANNELS * imageHeight * imageWidth;
  const numImagesStride = NUM_CHANNELS * imageHeight * imageWidth;
  const channelStride = imageHeight * imageWidth;
  
  for (let h = 0; h < imageHeight; h++) {
    for (let w = 0; w < imageWidth; w++) {
      const srcIdx = (h * imageWidth + w) * NUM_CHANNELS;
      const baseIdx = h * imageWidth + w;
      const batchOffset = BATCH_INDEX * batchStride;
      const numImagesOffset = NUM_IMAGES_INDEX * numImagesStride;
      
      // Channel 0 (R) - [1, 1, 0, h, w]
      const redOffset = batchOffset + numImagesOffset + CHANNEL_RED * channelStride + baseIdx;
      reshapedData[redOffset] = imageData[srcIdx + CHANNEL_RED];
      
      // Channel 1 (G) - [1, 1, 1, h, w]
      const greenOffset = batchOffset + numImagesOffset + CHANNEL_GREEN * channelStride + baseIdx;
      reshapedData[greenOffset] = imageData[srcIdx + CHANNEL_GREEN];
      
      // Channel 2 (B) - [1, 1, 2, h, w]
      const blueOffset = batchOffset + numImagesOffset + CHANNEL_BLUE * channelStride + baseIdx;
      reshapedData[blueOffset] = imageData[srcIdx + CHANNEL_BLUE];
    }
  }
  
  const imageTensorShape: ImageTensorShape = [
    IMAGE_BATCH_SIZE,
    IMAGE_NUM_IMAGES,
    NUM_CHANNELS,
    imageHeight,
    imageWidth
  ];
  const imageTensor = new ort.Tensor(
    IMAGE_TENSOR_DATA_TYPE,
    reshapedData,
    imageTensorShape
  );

  // Create pixel attention mask for vision encoder
  // The attention mask should match the spatial dimensions of the image
  // Shape: [batch_size, num_images, height, width] (rank 4)
  // For 512x512 image: [1, 1, 512, 512]
  const VISION_BATCH_SIZE = 1;
  const VISION_NUM_IMAGES = 1;
  const PIXEL_ATTENTION_MASK_ENABLED = 1;
  const PIXEL_ATTENTION_MASK_TYPE = 'bool';
  const attentionMaskData = new Uint8Array(VISION_BATCH_SIZE * VISION_NUM_IMAGES * imageHeight * imageWidth);
  // Fill with ones (all pixels should be attended to)
  // Use Uint8Array with values 0/1, which can be interpreted as boolean
  for (let i = 0; i < attentionMaskData.length; i++) {
    attentionMaskData[i] = PIXEL_ATTENTION_MASK_ENABLED;
  }
  const pixelAttentionMaskShape: PixelAttentionMaskShape = [
    VISION_BATCH_SIZE,
    VISION_NUM_IMAGES,
    imageHeight,
    imageWidth
  ];
  const pixelAttentionMask = new ort.Tensor(
    PIXEL_ATTENTION_MASK_TYPE,
    attentionMaskData,
    pixelAttentionMaskShape
  );

  // Run vision encoder to get image embeddings
  // ONNX Runtime Web uses Record<string, ort.Tensor> for inputs
  const visionInputs: Record<string, ort.Tensor> = {
    pixel_values: imageTensor,
    pixel_attention_mask: pixelAttentionMask,
  };
  
  const visionOutputs = await smolvlmModel.visionEncoder.run(visionInputs);
  
  // Get image embeddings from vision encoder output
  // Output name depends on model - check model outputs
  // Common names: 'last_hidden_state', 'image_embeds', 'pooler_output'
  let imageEmbeddings: ort.Tensor | undefined;
  const outputKeys = Object.keys(visionOutputs);
  if (outputKeys.length > 0) {
    const firstOutput = visionOutputs[outputKeys[0]];
    if (firstOutput instanceof ort.Tensor) {
      imageEmbeddings = firstOutput;
    }
  }
  
  if (!imageEmbeddings) {
    throw new Error('Vision encoder did not return embeddings');
  }
  
  if (onLog) {
    onLog(`Vision encoder output shape: [${imageEmbeddings.dims.join(', ')}]`, 'info');
    const imageEmbeddingsData = imageEmbeddings.data;
    if (imageEmbeddingsData instanceof Float32Array) {
      const sampleValues = imageEmbeddingsData.slice(0, 10);
      onLog(`Vision encoder output sample values (first 10): [${sampleValues.join(', ')}]`, 'info');
    }
  }

  // Prepare decoder inputs
  // Check what inputs the decoder supports
  const decoderInputNames = smolvlmModel.decoder.inputNames;
  const SUPPORTS_INPUT_IDS = decoderInputNames.includes('input_ids');
  const SUPPORTS_IMAGE_EMBEDS = decoderInputNames.includes('image_embeds') || decoderInputNames.includes('image_embeddings');
  
  // Get image embeddings dimensions
  const imageEmbeddingsDims = imageEmbeddings.dims;
  const BATCH_DIM_INDEX = 0;
  const SEQUENCE_DIM_INDEX = 1;
  const LAST_DIM_INDEX = imageEmbeddingsDims.length - 1;
  const DEFAULT_BATCH_SIZE = 1;
  const DEFAULT_SEQUENCE_LENGTH = 1;
  
  const decoderBatchSize = imageEmbeddingsDims[BATCH_DIM_INDEX] ?? DEFAULT_BATCH_SIZE;
  const imageSequenceLength = imageEmbeddingsDims[SEQUENCE_DIM_INDEX] ?? imageEmbeddingsDims[LAST_DIM_INDEX] ?? DEFAULT_SEQUENCE_LENGTH;
  const imageEmbeddingDim = imageEmbeddingsDims[imageEmbeddingsDims.length - 1] ?? 1;
  
  // Constants for generation loop
  const MAX_GENERATION_LENGTH = 128;
  const SINGLE_TOKEN_SEQUENCE_LENGTH = 1;
  
  // Get EOS token ID from tokenizer
  // Try to get from tokenizer config first (more reliable than encoding text)
  let EOS_TOKEN_ID: number | undefined;
  
  // Try to access tokenizer config if available
  // The @huggingface/tokenizers library doesn't expose config directly,
  // so we fall back to encoding the EOS text strings
  try {
    // Try <end_of_utterance> first (SmolVLM format)
    const eosEncoded = smolvlmModel.tokenizer.encode('<end_of_utterance>');
    if (eosEncoded.ids.length === 1) {
      // Only use if it's a single token (not multiple tokens)
      EOS_TOKEN_ID = eosEncoded.ids[0];
      if (onLog) {
        onLog(`EOS token ID from '<end_of_utterance>': ${EOS_TOKEN_ID}`, 'info');
      }
    }
  } catch {
    // Continue to next fallback
  }
  
  // Fallback to <|endoftext|> if <end_of_utterance> didn't work or wasn't single token
  if (EOS_TOKEN_ID === undefined) {
    try {
      const eosEncoded = smolvlmModel.tokenizer.encode('<|endoftext|>');
      if (eosEncoded.ids.length === 1) {
        EOS_TOKEN_ID = eosEncoded.ids[0];
        if (onLog) {
          onLog(`EOS token ID from '<|endoftext|>': ${EOS_TOKEN_ID}`, 'info');
        }
      }
    } catch {
      // Continue to final fallback
    }
  }
  
  // Final fallback to common EOS token ID (usually 2 for GPT-style models)
  if (EOS_TOKEN_ID === undefined) {
    EOS_TOKEN_ID = 2;
    if (onLog) {
      onLog(`Using default EOS token ID: ${EOS_TOKEN_ID}`, 'warning');
    }
  }
  
  // Check for both EOS markers in decoded text
  const EOS_TEXT_UTTERANCE = '<end_of_utterance>';
  const EOS_TEXT_ENDOFTEXT = '<|endoftext|>';
  
  // For the first iteration, we need to combine image embeddings with question tokens
  // The actual sequence length depends on how we provide the inputs:
  // - If using inputs_embeds only: sequence_length = imageSequenceLength
  // - If using image_embeds + input_ids: model may concatenate internally, so sequence_length = imageSequenceLength + questionTokenLength
  // - If using inputs_embeds + input_ids: model may concatenate internally, so sequence_length = imageSequenceLength + questionTokenLength
  const questionTokenLength = questionTokenIds.length;
  
  // Constants for decoder inputs
  const ATTENTION_MASK_ENABLED = BigInt(1);
  const DECODER_ATTENTION_MASK_TYPE = 'int64';
  const POSITION_IDS_TYPE = 'int64';
  
  // Helper function to create position IDs and attention mask for a given sequence length
  const createPositionAndAttention = (sequenceLength: number) => {
    // Create attention mask
    const attentionMaskData = new BigInt64Array(decoderBatchSize * sequenceLength);
    for (let i = 0; i < attentionMaskData.length; i++) {
      attentionMaskData[i] = ATTENTION_MASK_ENABLED;
    }
    const attentionMaskShape: DecoderAttentionMaskShape = [
      decoderBatchSize,
      sequenceLength
    ];
    const attentionMask = new ort.Tensor(
      DECODER_ATTENTION_MASK_TYPE,
      attentionMaskData,
      attentionMaskShape
    );
    
    // Create position IDs
    const positionIdsData = new BigInt64Array(decoderBatchSize * sequenceLength);
    for (let b = 0; b < decoderBatchSize; b++) {
      for (let s = 0; s < sequenceLength; s++) {
        positionIdsData[b * sequenceLength + s] = BigInt(s);
      }
    }
    const positionIdsShape: PositionIdsShape = [
      decoderBatchSize,
      sequenceLength
    ];
    const positionIds = new ort.Tensor(
      POSITION_IDS_TYPE,
      positionIdsData,
      positionIdsShape
    );
    
    return { attentionMask, positionIds, sequenceLength };
  };
  
  // Initialize past_key_values for autoregressive generation
  // For the first forward pass, past_key_values should be empty
  
  // Find all past_key_values input names (e.g., "past_key_values.0.key", "past_key_values.0.value", etc.)
  // Past key-value shape: [batch, num_heads, seq_len, head_dim] (rank 4)
  // For first pass: seq_len = 0, so shape is [batch, num_heads, 0, head_dim]
  // Model expects: num_heads=3, head_dim=64 (based on error message)
  const pastKeyValueInputs: Record<string, ort.Tensor> = {};
  const EMPTY_SEQUENCE_LENGTH = 0;
  const PAST_KEY_VALUE_DATA_TYPE = 'float32';
  // Model-specific values for SmolVLM-256M
  // These values are determined by the model architecture
  const PAST_KEY_VALUE_NUM_HEADS = 3;
  const PAST_KEY_VALUE_HEAD_DIM = 64;
  
  for (const inputName of decoderInputNames) {
    if (inputName.startsWith('past_key_values.')) {
      // For the first forward pass, create empty tensors with rank 4
      // Shape: [batch, num_heads, 0, head_dim]
      const pastKeyValueShape: PastKeyValueShape = [
        decoderBatchSize,
        PAST_KEY_VALUE_NUM_HEADS,
        EMPTY_SEQUENCE_LENGTH,
        PAST_KEY_VALUE_HEAD_DIM
      ];
      const totalElements = pastKeyValueShape.reduce((acc, dim) => acc * dim, 1);
      const emptyKeyValueData = new Float32Array(totalElements);
      pastKeyValueInputs[inputName] = new ort.Tensor(PAST_KEY_VALUE_DATA_TYPE, emptyKeyValueData, pastKeyValueShape);
    }
  }
  
  // Prepare initial decoder inputs
  // Critical: We cannot mix input_ids and inputs_embeds in the same ONNX call
  // For vision-language models, the typical approach is:
  // 1. If model supports image_embeds separately: use image_embeds + input_ids for question
  // 2. Otherwise: concatenate image embeddings with question embeddings in inputs_embeds
  // Since we don't have access to the embedding layer, we need to work with what we have
  
  let decoderInputs: Record<string, ort.Tensor>;
  let initialSequenceLength: number;
  let decoderAttentionMask: ort.Tensor;
  let positionIds: ort.Tensor;
  
  if (SUPPORTS_IMAGE_EMBEDS && SUPPORTS_INPUT_IDS && questionTokenLength > 0) {
    // Best case: Model supports both image_embeds and input_ids
    // Use image_embeds for image and input_ids for question tokens
    // The model may concatenate them internally, so sequence_length = imageSequenceLength + questionTokenLength
    initialSequenceLength = imageSequenceLength + questionTokenLength;
    const { attentionMask, positionIds: posIds } = createPositionAndAttention(initialSequenceLength);
    decoderAttentionMask = attentionMask;
    positionIds = posIds;
    
    const INPUT_IDS_DATA_TYPE = 'int64';
    const INPUT_IDS_BATCH_SIZE = 1;
    const inputIdsShape: InputIdsShape = [
      INPUT_IDS_BATCH_SIZE,
      questionTokenLength
    ];
    const inputIdsTensor = new ort.Tensor(
      INPUT_IDS_DATA_TYPE,
      new BigInt64Array(questionTokenIds.map(id => BigInt(id))),
      inputIdsShape
    );
    
    decoderInputs = {
      image_embeds: imageEmbeddings,
      input_ids: inputIdsTensor,
      attention_mask: decoderAttentionMask,
      position_ids: positionIds,
      ...pastKeyValueInputs,
    };
  } else if (smolvlmModel.embedTokens && questionTokenLength > 0) {
    // We have the embedding model - use it to convert question tokens to embeddings
    // Then concatenate with image embeddings
    const INPUT_IDS_DATA_TYPE = 'int64';
    const INPUT_IDS_BATCH_SIZE = 1;
    const inputIdsShape: InputIdsShape = [
      INPUT_IDS_BATCH_SIZE,
      questionTokenLength
    ];
    const inputIdsTensor = new ort.Tensor(
      INPUT_IDS_DATA_TYPE,
      new BigInt64Array(questionTokenIds.map(id => BigInt(id))),
      inputIdsShape
    );
    
    // Convert question token IDs to embeddings using embed_tokens model
    // Check if embedding model needs attention_mask
    const embedInputNames = smolvlmModel.embedTokens.inputNames;
    const embedNeedsAttentionMask = embedInputNames.includes('attention_mask');
    
    const embedInputs: Record<string, ort.Tensor> = { input_ids: inputIdsTensor };
    
    // Add attention_mask if required by embedding model
    if (embedNeedsAttentionMask) {
      const embedAttentionMaskData = new BigInt64Array(questionTokenLength);
      for (let i = 0; i < questionTokenLength; i++) {
        embedAttentionMaskData[i] = BigInt(1);
      }
      const embedAttentionMask = new ort.Tensor(
        'int64',
        embedAttentionMaskData,
        [1, questionTokenLength]
      );
      embedInputs.attention_mask = embedAttentionMask;
    }
    
    const embedOutputs = await smolvlmModel.embedTokens.run(embedInputs);
    const questionEmbeddingsTensor = embedOutputs[Object.keys(embedOutputs)[0]];
    if (!(questionEmbeddingsTensor instanceof ort.Tensor)) {
      throw new Error('Embedding model did not return valid embeddings');
    }
    
    // Conditional merge: Replace <image> token embedding with image embeddings
    // This is the correct approach per technical guidance - image embeddings replace
    // the single <image> token's embedding within the tokenized text sequence
    const imageEmbedsDims = imageEmbeddings.dims;
    const questionEmbedsDims = questionEmbeddingsTensor.dims;
    const imageBatch = imageEmbedsDims[0] ?? 1;
    const imageSeqLen = imageEmbedsDims[1] ?? 1; // ~64 for 512x512 image
    const questionSeqLen = questionEmbedsDims[1] ?? 1; // Full prompt length (includes <image> token)
    const embeddingDim = imageEmbedsDims[imageEmbedsDims.length - 1] ?? questionEmbedsDims[questionEmbedsDims.length - 1] ?? 576;
    
    if (onLog) {
      onLog(`Conditional merge: questionSeqLen=${questionSeqLen}, imageSeqLen=${imageSeqLen}, imageTokenIndex=${imageTokenIndex}, embeddingDim=${embeddingDim}`, 'info');
    }
    
    // Final sequence length: (questionSeqLen - 1) + imageSeqLen
    // Subtract 1 because we're replacing the single <image> token with imageSeqLen tokens
    const totalSeqLen = (questionSeqLen - 1) + imageSeqLen;
    const mergedData = new Float32Array(imageBatch * totalSeqLen * embeddingDim);
    
    if (onLog) {
      onLog(`Conditional merge: totalSeqLen=${totalSeqLen}, mergedData.length=${mergedData.length}`, 'info');
    }
    
    // Get data arrays
    const imageData = imageEmbeddings.data;
    const questionData = questionEmbeddingsTensor.data;
    
    if (!(imageData instanceof Float32Array) || !(questionData instanceof Float32Array)) {
      throw new Error('Image or question embeddings data is not Float32Array');
    }
    
    // Conditional merge: Copy question embeddings up to <image> token, replace <image> token
    // with image embeddings, then copy remaining question embeddings after image embeddings
    // Data layout: [batch0_seq0_emb0, batch0_seq0_emb1, ..., batch0_seq0_embN, batch0_seq1_emb0, ...]
    const embeddingDimSize = embeddingDim;
    const questionElementsPerBatch = questionSeqLen * embeddingDimSize;
    const imageElementsPerBatch = imageSeqLen * embeddingDimSize;
    const totalElementsPerBatch = totalSeqLen * embeddingDimSize;
    
    for (let b = 0; b < imageBatch; b++) {
      const questionBatchStart = b * questionElementsPerBatch;
      const imageBatchStart = b * imageElementsPerBatch;
      const mergedBatchStart = b * totalElementsPerBatch;
      
      // Copy tokens before <image> token
      // beforeImageLen: number of tokens before <image> token (e.g., 4)
      const beforeImageLen = imageTokenIndex;
      const beforeImageStart = questionBatchStart;
      const beforeImageEnd = beforeImageStart + beforeImageLen * embeddingDimSize;
      
      if (onLog && b === 0) {
        onLog(`Conditional merge batch ${b}: beforeImageLen=${beforeImageLen}, beforeImageStart=${beforeImageStart}, beforeImageEnd=${beforeImageEnd}`, 'info');
      }
      
      mergedData.set(
        questionData.subarray(beforeImageStart, beforeImageEnd),
        mergedBatchStart
      );
      
      // Replace <image> token with image embeddings
      // mergedImageStart: position in mergedData where image embeddings start
      const mergedImageStart = mergedBatchStart + beforeImageLen * embeddingDimSize;
      
      if (onLog && b === 0) {
        onLog(`Conditional merge batch ${b}: mergedImageStart=${mergedImageStart}, imageBatchStart=${imageBatchStart}, imageElementsPerBatch=${imageElementsPerBatch}`, 'info');
      }
      
      mergedData.set(
        imageData.subarray(imageBatchStart, imageBatchStart + imageElementsPerBatch),
        mergedImageStart
      );
      
      // Copy tokens after <image> token
      // Skip the <image> token embedding (one embeddingDimSize worth of data)
      // afterImageStart: index in questionData where tokens after <image> start
      // This is afterImageEnd of the <image> token, which is beforeImageEnd + embeddingDimSize
      const afterImageStart = beforeImageEnd + embeddingDimSize;
      const afterImageLen = questionSeqLen - imageTokenIndex - 1;
      const mergedAfterStart = mergedImageStart + imageElementsPerBatch;
      
      if (onLog && b === 0) {
        onLog(`Conditional merge batch ${b}: afterImageStart=${afterImageStart}, afterImageLen=${afterImageLen}, mergedAfterStart=${mergedAfterStart}`, 'info');
      }
      
      // Verify we're not going out of bounds
      const afterImageEnd = afterImageStart + afterImageLen * embeddingDimSize;
      if (afterImageEnd > questionData.length) {
        throw new Error(`Conditional merge: afterImageEnd (${afterImageEnd}) exceeds questionData length (${questionData.length})`);
      }
      if (mergedAfterStart + afterImageLen * embeddingDimSize > mergedData.length) {
        throw new Error(`Conditional merge: mergedAfterStart + afterImageLen (${mergedAfterStart + afterImageLen * embeddingDimSize}) exceeds mergedData length (${mergedData.length})`);
      }
      
      mergedData.set(
        questionData.subarray(afterImageStart, afterImageEnd),
        mergedAfterStart
      );
      
      if (onLog && b === 0) {
        onLog(`Conditional merge batch ${b}: Successfully merged. Final mergedData length: ${mergedData.length}`, 'info');
      }
    }
    
    const mergedEmbeddings = new ort.Tensor(
      'float32',
      mergedData,
      [imageBatch, totalSeqLen, embeddingDim]
    );
    
    if (onLog) {
      onLog(`Conditional merge: questionSeqLen=${questionSeqLen}, imageSeqLen=${imageSeqLen}, totalSeqLen=${totalSeqLen}`, 'info');
      onLog(`Merged embeddings shape: [${imageBatch}, ${totalSeqLen}, ${embeddingDim}]`, 'info');
      const mergedSampleValues = mergedData.slice(0, 20);
      onLog(`Merged embeddings sample values (first 20): [${mergedSampleValues.join(', ')}]`, 'info');
      // Also check values at the image embedding position
      const imageStartInMerged = imageTokenIndex * embeddingDim;
      const imageSampleValues = mergedData.slice(imageStartInMerged, imageStartInMerged + 20);
      onLog(`Merged embeddings at image position (indices ${imageStartInMerged}-${imageStartInMerged + 19}): [${imageSampleValues.join(', ')}]`, 'info');
    }
    
    initialSequenceLength = totalSeqLen;
    const { attentionMask, positionIds: posIds } = createPositionAndAttention(initialSequenceLength);
    decoderAttentionMask = attentionMask;
    positionIds = posIds;
    
    decoderInputs = {
      inputs_embeds: mergedEmbeddings,
      attention_mask: decoderAttentionMask,
      position_ids: positionIds,
      ...pastKeyValueInputs,
    };
  } else if (SUPPORTS_INPUT_IDS && questionTokenLength > 0) {
    // Model supports input_ids but not image_embeds, and we don't have embedTokens
    // Try mixing inputs_embeds (image) + input_ids (question)
    // The model may concatenate them internally, so sequence_length = imageSequenceLength + questionTokenLength
    initialSequenceLength = imageSequenceLength + questionTokenLength;
    const { attentionMask, positionIds: posIds } = createPositionAndAttention(initialSequenceLength);
    decoderAttentionMask = attentionMask;
    positionIds = posIds;
    
    const INPUT_IDS_DATA_TYPE = 'int64';
    const INPUT_IDS_BATCH_SIZE = 1;
    const inputIdsShape: InputIdsShape = [
      INPUT_IDS_BATCH_SIZE,
      questionTokenLength
    ];
    const inputIdsTensor = new ort.Tensor(
      INPUT_IDS_DATA_TYPE,
      new BigInt64Array(questionTokenIds.map(id => BigInt(id))),
      inputIdsShape
    );
    
    // Try providing both - some models can handle this
    // If the model rejects this, we'll need to concatenate embeddings manually
    decoderInputs = {
      inputs_embeds: imageEmbeddings,
      input_ids: inputIdsTensor,
      attention_mask: decoderAttentionMask,
      position_ids: positionIds,
      ...pastKeyValueInputs,
    };
    
    // Note: This may fail if the model doesn't allow mixing input_ids and inputs_embeds
    // In that case, we'll need to extract the embedding layer or use a different approach
  } else {
    // Model only supports inputs_embeds (or no question tokens)
    // Use inputs_embeds with image embeddings only
    // Sequence length is just the image sequence length
    initialSequenceLength = imageSequenceLength;
    const { attentionMask, positionIds: posIds } = createPositionAndAttention(initialSequenceLength);
    decoderAttentionMask = attentionMask;
    positionIds = posIds;
    
    decoderInputs = {
      inputs_embeds: imageEmbeddings,
      attention_mask: decoderAttentionMask,
      position_ids: positionIds,
      ...pastKeyValueInputs,
    };
  }

  // Autoregressive generation loop
  const generatedTokenIds: number[] = [];
  let currentPastKeyValues: Record<string, ort.Tensor> = { ...pastKeyValueInputs };
  let shouldStopGeneration = false;

  if (onLog) {
    onLog(`Starting autoregressive generation. Initial sequence length: ${initialSequenceLength}, Max steps: ${MAX_GENERATION_LENGTH}`, 'info');
    const decoderInputKeys = Object.keys(decoderInputs);
    onLog(`Decoder input keys: ${decoderInputKeys.join(', ')}`, 'info');
    if (decoderInputs.inputs_embeds) {
      onLog(`Initial inputs_embeds shape: [${decoderInputs.inputs_embeds.dims.join(', ')}]`, 'info');
    }
    if (decoderInputs.input_ids) {
      onLog(`Initial input_ids shape: [${decoderInputs.input_ids.dims.join(', ')}]`, 'info');
    }
  }

  for (let step = 0; step < MAX_GENERATION_LENGTH && !shouldStopGeneration; step++) {
    if (onLog && step === 0) {
      onLog(`Generation step ${step + 1}/${MAX_GENERATION_LENGTH}`, 'info');
    }
    
    // Run decoder
    const decoderOutputs = await smolvlmModel.decoder.run(decoderInputs);
    
    // Extract logits from decoder output
    let logits: ort.Tensor | undefined;
    const decoderOutputKeys = Object.keys(decoderOutputs);
    
    if (onLog && step === 0) {
      onLog(`Decoder output keys: ${decoderOutputKeys.join(', ')}`, 'info');
    }
    
    const FIRST_DECODER_OUTPUT_INDEX = 0;
    const MIN_DECODER_OUTPUTS_REQUIRED = 1;
    if (decoderOutputKeys.length >= MIN_DECODER_OUTPUTS_REQUIRED) {
      // First, explicitly look for output named 'logits'
      if ('logits' in decoderOutputs) {
        const logitsOutput = decoderOutputs.logits;
        if (logitsOutput instanceof ort.Tensor) {
          logits = logitsOutput;
          if (onLog && step === 0) {
            onLog(`Found logits output with shape [${logitsOutput.dims.join(', ')}]`, 'info');
          }
        }
      }
      
      // If not found by name, find by shape: logits should be [batch, seq_len, vocab_size]
      if (!logits) {
        for (const key of decoderOutputKeys) {
          const output = decoderOutputs[key];
          if (output instanceof ort.Tensor && !key.startsWith('past_key_values')) {
            const outputDims = output.dims;
            // Logits should have at least 2 dimensions, and last dimension should be vocab_size
            if (outputDims.length >= 2) {
              logits = output;
              if (onLog && step === 0) {
                onLog(`Found logits in output "${key}" with shape [${outputDims.join(', ')}]`, 'info');
              }
              break;
            }
          }
        }
      }
      
      // Fallback to first output if no logits found
      if (!logits && decoderOutputKeys.length > 0) {
        const firstOutput = decoderOutputs[decoderOutputKeys[FIRST_DECODER_OUTPUT_INDEX]];
        if (firstOutput instanceof ort.Tensor) {
          logits = firstOutput;
          if (onLog && step === 0) {
            onLog(`Using first output as logits with shape [${firstOutput.dims.join(', ')}]`, 'warning');
          }
        }
      }
    }
    
    if (!logits) {
      throw new Error('Decoder did not return logits');
    }
    
    if (onLog && step === 0) {
      onLog(`Logits shape: [${logits.dims.join(', ')}]`, 'info');
    }

    // Extract past_key_values from outputs for next iteration
    // Ensure ALL required past_key_values are present (from decoder.inputNames)
    const newPastKeyValues: Record<string, ort.Tensor> = {};
    
    // First, extract past_key_values from decoder outputs
    for (const key of decoderOutputKeys) {
      if (key.startsWith('past_key_values.')) {
        const pastValue = decoderOutputs[key];
        if (pastValue instanceof ort.Tensor) {
          newPastKeyValues[key] = pastValue;
        }
      }
    }
    
    // Ensure all required past_key_values inputs are present
    // If a past_key_value is missing from outputs, keep the previous one or create empty
    for (const inputName of decoderInputNames) {
      if (inputName.startsWith('past_key_values.')) {
        if (!(inputName in newPastKeyValues)) {
          // If not in outputs, use the previous value or create empty
          if (inputName in currentPastKeyValues) {
            newPastKeyValues[inputName] = currentPastKeyValues[inputName];
          } else {
            // Create empty past_key_value as fallback
            // Use the shape from the first past_key_value if available, otherwise use defaults
            const pastKeyValueShape: PastKeyValueShape = [
              decoderBatchSize,
              PAST_KEY_VALUE_NUM_HEADS,
              EMPTY_SEQUENCE_LENGTH,
              PAST_KEY_VALUE_HEAD_DIM
            ];
            const totalElements = pastKeyValueShape.reduce((acc, dim) => acc * dim, 1);
            const emptyKeyValueData = new Float32Array(totalElements);
            newPastKeyValues[inputName] = new ort.Tensor(PAST_KEY_VALUE_DATA_TYPE, emptyKeyValueData, pastKeyValueShape);
          }
        }
      }
    }

    // Get predicted token ID from logits (argmax of last position)
    const logitsDataRaw = logits.data;
    let logitsData: Float32Array;
    if (logitsDataRaw instanceof Float32Array) {
      logitsData = logitsDataRaw;
    } else {
      // Convert to Float32Array if needed
      const values: number[] = [];
      const lengthDescriptor = Object.getOwnPropertyDescriptor(logitsDataRaw, 'length');
      if (lengthDescriptor && typeof lengthDescriptor.value === 'number') {
        const length = lengthDescriptor.value;
        for (let i = 0; i < length; i++) {
          const itemDescriptor = Object.getOwnPropertyDescriptor(logitsDataRaw, i.toString());
          if (itemDescriptor && typeof itemDescriptor.value === 'number') {
            values.push(itemDescriptor.value);
          }
        }
      }
      logitsData = new Float32Array(values);
    }
    
    const logitsDims = logits.dims;
    const VOCAB_SIZE_DIM_INDEX = logitsDims.length - 1;
    const SEQUENCE_LENGTH_DIM_INDEX = logitsDims.length - 2;
    const LAST_POSITION_OFFSET = 1;
    const ARGMAX_START_INDEX = 0;
    const ARGMAX_LOOP_START = 1;
    
    const vocabSize = logitsDims[VOCAB_SIZE_DIM_INDEX];
    const sequenceLength = logitsDims[SEQUENCE_LENGTH_DIM_INDEX];
    
    // For first iteration: sequenceLength includes input tokens (image + question)
    // For subsequent iterations: sequenceLength should be 1 (single new token)
    // Always use the last position in the sequence for argmax
    const lastPositionIdx = (sequenceLength - LAST_POSITION_OFFSET) * vocabSize;
    
    if (onLog && (step === 0 || step % 10 === 0)) {
      onLog(`Step ${step + 1}: Logits shape [${logitsDims.join(', ')}], Sequence length: ${sequenceLength}, Vocab size: ${vocabSize}, Using position index: ${lastPositionIdx}`, 'info');
    }
    
    // Validate position index
    if (lastPositionIdx < 0 || lastPositionIdx + vocabSize > logitsData.length) {
      const errorMsg = `Invalid position index: ${lastPositionIdx} (logits data length: ${logitsData.length}, vocab size: ${vocabSize})`;
      if (onLog) {
        onLog(errorMsg, 'error');
      }
      throw new Error(errorMsg);
    }
    
    let maxIdx = ARGMAX_START_INDEX;
    let maxVal = logitsData[lastPositionIdx];
    for (let i = ARGMAX_LOOP_START; i < vocabSize; i++) {
      const val = logitsData[lastPositionIdx + i];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = i;
      }
    }
    
    // Check for EOS token
    if (EOS_TOKEN_ID !== undefined && maxIdx === EOS_TOKEN_ID) {
      if (onLog) {
        onLog(`Step ${step + 1}: EOS token detected (ID: ${EOS_TOKEN_ID}), stopping generation`, 'info');
      }
      shouldStopGeneration = true;
      break;
    }
    
    // Decode current token to check if it's EOS text
    const currentTokenText = smolvlmModel.tokenizer.decode([maxIdx]);
    if (currentTokenText.includes(EOS_TEXT_UTTERANCE) || currentTokenText.includes(EOS_TEXT_ENDOFTEXT)) {
      if (onLog) {
        onLog(`Step ${step + 1}: EOS text detected ("${currentTokenText}"), stopping generation`, 'info');
      }
      shouldStopGeneration = true;
      break;
    }
    
    // Check for repetition: if we've generated the same token multiple times in a row, stop
    const REPETITION_THRESHOLD = 2; // Lowered from 3 to catch repetition earlier
    if (generatedTokenIds.length >= REPETITION_THRESHOLD && !shouldStopGeneration) {
      const lastTokens = generatedTokenIds.slice(-REPETITION_THRESHOLD);
      const allSame = lastTokens.every(tokenId => tokenId === maxIdx);
      if (allSame) {
        // Same token repeated - likely stuck in a loop
        if (onLog) {
          onLog(`Step ${step + 1}: Repetition detected (token ${maxIdx} repeated ${REPETITION_THRESHOLD + 1} times), stopping generation`, 'warning');
        }
        shouldStopGeneration = true;
        break;
      }
    }
    
    // Check if we should stop before adding token
    if (shouldStopGeneration) {
      break;
    }
    
    // Check for longer repetitive patterns (n-grams)
    // Check for 2-gram repetition
    if (generatedTokenIds.length >= 4 && !shouldStopGeneration) {
      const lastFour = generatedTokenIds.slice(-4);
      const pattern1 = lastFour.slice(0, 2);
      const pattern2 = lastFour.slice(2, 4);
      if (pattern1.every((val, idx) => val === pattern2[idx])) {
        if (onLog) {
          onLog(`Step ${step + 1}: Pattern repetition detected (2-token pattern repeating), stopping generation`, 'warning');
        }
        shouldStopGeneration = true;
      }
    }
    
    // Check for 3-gram repetition
    if (generatedTokenIds.length >= 6 && !shouldStopGeneration) {
      const lastSix = generatedTokenIds.slice(-6);
      const pattern1 = lastSix.slice(0, 3);
      const pattern2 = lastSix.slice(3, 6);
      if (pattern1.every((val, idx) => val === pattern2[idx])) {
        if (onLog) {
          onLog(`Step ${step + 1}: Pattern repetition detected (3-token pattern repeating), stopping generation`, 'warning');
        }
        shouldStopGeneration = true;
      }
    }
    
    // Check for 5-gram repetition (10 tokens total) - catches longer patterns
    if (generatedTokenIds.length >= 10 && !shouldStopGeneration) {
      const lastTen = generatedTokenIds.slice(-10);
      const pattern1 = lastTen.slice(0, 5);
      const pattern2 = lastTen.slice(5, 10);
      if (pattern1.every((val, idx) => val === pattern2[idx])) {
        if (onLog) {
          onLog(`Step ${step + 1}: Pattern repetition detected (5-token pattern repeating), stopping generation`, 'warning');
        }
        shouldStopGeneration = true;
      }
    }
    
    // Check for 10-gram repetition (20 tokens total) - catches very long patterns
    if (generatedTokenIds.length >= 20 && !shouldStopGeneration) {
      const lastTwenty = generatedTokenIds.slice(-20);
      const pattern1 = lastTwenty.slice(0, 10);
      const pattern2 = lastTwenty.slice(10, 20);
      if (pattern1.every((val, idx) => val === pattern2[idx])) {
        if (onLog) {
          onLog(`Step ${step + 1}: Pattern repetition detected (10-token pattern repeating), stopping generation`, 'warning');
        }
        shouldStopGeneration = true;
      }
    }
    
    // Check if we should stop before processing this token
    if (shouldStopGeneration) {
      break;
    }
    
    // Sliding window check: check if any 10-token pattern repeats in last 30 tokens
    // This catches patterns that don't align perfectly at boundaries
    if (generatedTokenIds.length >= 30 && !shouldStopGeneration) {
      const lastThirty = generatedTokenIds.slice(-30);
      for (let i = 0; i <= 10 && !shouldStopGeneration; i++) {
        const pattern = lastThirty.slice(i, i + 10);
        const nextOccurrence = lastThirty.slice(i + 10, i + 20);
        if (pattern.length === 10 && nextOccurrence.length === 10 &&
            pattern.every((val, idx) => val === nextOccurrence[idx])) {
          if (onLog) {
            onLog(`Step ${step + 1}: Sliding window repetition detected (10-token pattern at offset ${i} repeating), stopping generation`, 'warning');
          }
          shouldStopGeneration = true;
          break;
        }
      }
    }
    
    // Early pattern detection: if same 5-token sequence appears 3+ times in last 30 tokens, stop
    if (generatedTokenIds.length >= 30 && !shouldStopGeneration) {
      const lastThirty = generatedTokenIds.slice(-30);
      for (let i = 0; i <= 20 && !shouldStopGeneration; i++) {
        const pattern = lastThirty.slice(i, i + 5);
        if (pattern.length === 5) {
          let occurrenceCount = 1; // Count the pattern itself
          for (let j = i + 5; j <= 25 && !shouldStopGeneration; j++) {
            const candidate = lastThirty.slice(j, j + 5);
            if (candidate.length === 5 && pattern.every((val, idx) => val === candidate[idx])) {
              occurrenceCount++;
              if (occurrenceCount >= 3) {
                if (onLog) {
                  onLog(`Step ${step + 1}: Early pattern detection (5-token pattern appears ${occurrenceCount} times), stopping generation`, 'warning');
                }
                shouldStopGeneration = true;
                break;
              }
            }
          }
        }
      }
    }
    
    // Check if we should stop before processing this token
    if (shouldStopGeneration) {
      break;
    }
    
    // Check for empty or invalid token
    if (maxIdx < 0 || maxIdx >= vocabSize) {
      if (onLog) {
        onLog(`Step ${step + 1}: Invalid token ID ${maxIdx} (vocab size: ${vocabSize}), stopping generation`, 'error');
      }
      break;
    }
    
    // Check if token decodes to empty string (might indicate issue)
    const decodedToken = smolvlmModel.tokenizer.decode([maxIdx]);
    if (decodedToken.trim().length === 0 && generatedTokenIds.length > 0) {
      if (onLog) {
        onLog(`Step ${step + 1}: Generated empty token (ID: ${maxIdx}), stopping generation`, 'warning');
      }
      break;
    }
    
    // Maximum repetition count: if any token appears > 50% of the time in last 20 tokens, stop
    if (generatedTokenIds.length >= 20) {
      const lastTwenty = generatedTokenIds.slice(-20);
      const tokenCounts = new Map<number, number>();
      for (const tokenId of lastTwenty) {
        tokenCounts.set(tokenId, (tokenCounts.get(tokenId) || 0) + 1);
      }
      for (const [tokenId, count] of tokenCounts.entries()) {
        if (count > 10) { // More than 50% of 20 tokens
          if (onLog) {
            onLog(`Step ${step + 1}: Maximum repetition count detected (token ${tokenId} appears ${count}/20 times), stopping generation`, 'warning');
          }
          break;
        }
      }
      // Check if the new token would exceed the threshold
      const newTokenCount = (tokenCounts.get(maxIdx) || 0) + 1;
      if (newTokenCount > 10) {
        if (onLog) {
          onLog(`Step ${step + 1}: Maximum repetition count would be exceeded (token ${maxIdx} would appear ${newTokenCount}/20 times), stopping generation`, 'warning');
        }
        break;
      }
    }
    
    // Pattern frequency: if same 5-token pattern appears 3+ times in last 30 tokens, stop immediately
    if (generatedTokenIds.length >= 30) {
      const lastThirty = generatedTokenIds.slice(-30);
      for (let i = 0; i <= 20; i++) {
        const pattern = lastThirty.slice(i, i + 5);
        if (pattern.length === 5) {
          let occurrenceCount = 1;
          for (let j = i + 5; j <= 25; j++) {
            const candidate = lastThirty.slice(j, j + 5);
            if (candidate.length === 5 && pattern.every((val, idx) => val === candidate[idx])) {
              occurrenceCount++;
            }
          }
          if (occurrenceCount >= 3) {
            if (onLog) {
              onLog(`Step ${step + 1}: Pattern frequency detected (5-token pattern appears ${occurrenceCount} times), stopping generation`, 'warning');
            }
            break;
          }
        }
      }
    }
    
    // Low diversity check: if vocabulary diversity in last 20 tokens is very low (< 5 unique tokens), stop
    if (generatedTokenIds.length >= 20) {
      const lastTwenty = generatedTokenIds.slice(-20);
      const uniqueTokens = new Set(lastTwenty);
      if (uniqueTokens.size < 5) {
        if (onLog) {
          onLog(`Step ${step + 1}: Low diversity detected (only ${uniqueTokens.size} unique tokens in last 20), stopping generation`, 'warning');
        }
        break;
      }
      // Check if adding the new token would reduce diversity too much
      const newUniqueCount = new Set([...lastTwenty, maxIdx]).size;
      if (newUniqueCount < 5 && lastTwenty.length >= 20) {
        if (onLog) {
          onLog(`Step ${step + 1}: Low diversity would be exceeded (would have ${newUniqueCount} unique tokens), stopping generation`, 'warning');
        }
        break;
      }
    }
    
    // Add generated token to sequence
    generatedTokenIds.push(maxIdx);
    
    if (onLog && (step === 0 || step % 5 === 0 || generatedTokenIds.length % 10 === 0)) {
      const partialText = smolvlmModel.tokenizer.decode(generatedTokenIds.slice(-10));
      onLog(`Step ${step + 1}: Generated ${generatedTokenIds.length} tokens so far. Last 10 tokens: "${partialText}"`, 'info');
    }
    
    // Update past_key_values for next iteration
    currentPastKeyValues = newPastKeyValues;
    
    // Prepare inputs for next iteration
    // For subsequent steps, use input_ids for the new token if supported
    // Otherwise, we need to extract the embedding from the logits or use a different approach
    
    // Fix position IDs: use absolute position in full sequence
    // For subsequent iterations (step > 0), position ID should be the current sequence length
    // which is: initialSequenceLength (input tokens) + generatedTokenIds.length (generated tokens so far)
    // Note: generatedTokenIds.length already includes the token we just added, so this is correct
    const nextPosition = initialSequenceLength + generatedTokenIds.length;
    
    if (onLog && step < 3) {
      onLog(`Step ${step + 1}: Next position ID will be ${nextPosition} (initialSeqLen=${initialSequenceLength}, generatedTokens=${generatedTokenIds.length})`, 'info');
    }
    
    // Update position IDs for next token
    const nextPositionIdsData = new BigInt64Array(decoderBatchSize * SINGLE_TOKEN_SEQUENCE_LENGTH);
    for (let b = 0; b < decoderBatchSize; b++) {
      nextPositionIdsData[b] = BigInt(nextPosition);
    }
    const nextPositionIds = new ort.Tensor(
      POSITION_IDS_TYPE,
      nextPositionIdsData,
      [decoderBatchSize, SINGLE_TOKEN_SEQUENCE_LENGTH]
    );
    
    // Update attention mask
    const nextAttentionMaskData = new BigInt64Array(decoderBatchSize * SINGLE_TOKEN_SEQUENCE_LENGTH);
    for (let i = 0; i < nextAttentionMaskData.length; i++) {
      nextAttentionMaskData[i] = ATTENTION_MASK_ENABLED;
    }
    const nextAttentionMask = new ort.Tensor(
      DECODER_ATTENTION_MASK_TYPE,
      nextAttentionMaskData,
      [decoderBatchSize, SINGLE_TOKEN_SEQUENCE_LENGTH]
    );
    
    // For subsequent iterations, use input_ids if supported (better than random embeddings)
    if (SUPPORTS_INPUT_IDS) {
      // Use input_ids for the new token
      const INPUT_IDS_DATA_TYPE = 'int64';
      const INPUT_IDS_BATCH_SIZE = 1;
      const nextInputIdsShape: InputIdsShape = [
        INPUT_IDS_BATCH_SIZE,
        SINGLE_TOKEN_SEQUENCE_LENGTH
      ];
      const nextInputIds = new ort.Tensor(
        INPUT_IDS_DATA_TYPE,
        new BigInt64Array([BigInt(maxIdx)]),
        nextInputIdsShape
      );
      
      // Update decoder inputs for next iteration
      decoderInputs = {
        input_ids: nextInputIds,
        attention_mask: nextAttentionMask,
        position_ids: nextPositionIds,
        ...currentPastKeyValues,
      };
      
      // Remove inputs_embeds if it was present
      if ('inputs_embeds' in decoderInputs) {
        delete decoderInputs.inputs_embeds;
      }
    } else if (smolvlmModel.embedTokens) {
      // If input_ids not supported but we have embedTokens, use it to convert token ID to embedding
      const INPUT_IDS_DATA_TYPE = 'int64';
      const INPUT_IDS_BATCH_SIZE = 1;
      const nextInputIdsShape: InputIdsShape = [
        INPUT_IDS_BATCH_SIZE,
        SINGLE_TOKEN_SEQUENCE_LENGTH
      ];
      const nextInputIds = new ort.Tensor(
        INPUT_IDS_DATA_TYPE,
        new BigInt64Array([BigInt(maxIdx)]),
        nextInputIdsShape
      );
      
      // Convert token ID to embedding using embedTokens model
      const embedInputNames = smolvlmModel.embedTokens.inputNames;
      const embedNeedsAttentionMask = embedInputNames.includes('attention_mask');
      const embedInputs: Record<string, ort.Tensor> = { input_ids: nextInputIds };
      
      if (embedNeedsAttentionMask) {
        const embedAttentionMaskData = new BigInt64Array(SINGLE_TOKEN_SEQUENCE_LENGTH);
        embedAttentionMaskData[0] = ATTENTION_MASK_ENABLED;
        const embedAttentionMask = new ort.Tensor(
          DECODER_ATTENTION_MASK_TYPE,
          embedAttentionMaskData,
          [INPUT_IDS_BATCH_SIZE, SINGLE_TOKEN_SEQUENCE_LENGTH]
        );
        embedInputs.attention_mask = embedAttentionMask;
      }
      
      const embedOutputs = await smolvlmModel.embedTokens.run(embedInputs);
      const tokenEmbeddingTensor = embedOutputs[Object.keys(embedOutputs)[0]];
      if (!(tokenEmbeddingTensor instanceof ort.Tensor)) {
        throw new Error('Embedding model did not return valid embedding for token');
      }
      
      // Verify embedding dimension matches expected dimension
      const tokenEmbeddingDims = tokenEmbeddingTensor.dims;
      const tokenEmbeddingDim = tokenEmbeddingDims[tokenEmbeddingDims.length - 1];
      
      if (onLog && step === 0) {
        onLog(`Token embedding shape: [${tokenEmbeddingDims.join(', ')}], Expected embedding dim: ${imageEmbeddingDim}`, 'info');
      }
      
      // If dimensions don't match, log a warning but continue (model may handle it)
      if (tokenEmbeddingDim !== imageEmbeddingDim) {
        if (onLog) {
          onLog(`Warning: Token embedding dimension (${tokenEmbeddingDim}) doesn't match image embedding dimension (${imageEmbeddingDim})`, 'warning');
        }
      }
      
      // Update decoder inputs for next iteration
      decoderInputs = {
        inputs_embeds: tokenEmbeddingTensor,
        attention_mask: nextAttentionMask,
        position_ids: nextPositionIds,
        ...currentPastKeyValues,
      };
    } else {
      // Fallback: If input_ids not supported and no embedTokens, use zero embeddings
      // This is not ideal but may work if past_key_values carry enough context
      const EMBEDDING_DATA_TYPE = 'float32';
      const newTokenEmbeddingData = new Float32Array(decoderBatchSize * SINGLE_TOKEN_SEQUENCE_LENGTH * imageEmbeddingDim);
      // Use zeros - the past_key_values should provide the necessary context
      for (let i = 0; i < newTokenEmbeddingData.length; i++) {
        newTokenEmbeddingData[i] = 0;
      }
      const newTokenEmbeddingShape = [
        decoderBatchSize,
        SINGLE_TOKEN_SEQUENCE_LENGTH,
        imageEmbeddingDim
      ];
      const newTokenEmbedding = new ort.Tensor(
        EMBEDDING_DATA_TYPE,
        newTokenEmbeddingData,
        newTokenEmbeddingShape
      );
      
      // Update decoder inputs for next iteration
      decoderInputs = {
        inputs_embeds: newTokenEmbedding,
        attention_mask: nextAttentionMask,
        position_ids: nextPositionIds,
        ...currentPastKeyValues,
      };
    }
  }
  
  // Decode all generated tokens
  if (generatedTokenIds.length === 0) {
    return '';
  }
  
  const generatedText = smolvlmModel.tokenizer.decode(generatedTokenIds);
  
  // Remove any trailing EOS tokens from the decoded text
  let cleanedText = generatedText.trim();
  // Remove <end_of_utterance> markers
  while (cleanedText.endsWith(EOS_TEXT_UTTERANCE)) {
    cleanedText = cleanedText.slice(0, -EOS_TEXT_UTTERANCE.length).trim();
  }
  // Remove <|endoftext|> markers
  while (cleanedText.endsWith(EOS_TEXT_ENDOFTEXT)) {
    cleanedText = cleanedText.slice(0, -EOS_TEXT_ENDOFTEXT.length).trim();
  }
  
  return cleanedText;
}

/**
 * Check if SmolVLM-256M model is loaded
 */
export function isSmolVLM256MLoaded(): boolean {
  return smolvlmModel.isLoaded;
}

