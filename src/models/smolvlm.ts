// SmolVLM-500M integration using ONNX Runtime Web
// Handles model loading, tokenization, chat template, and inference

import * as ort from 'onnxruntime-web';

// SmolVLM model configuration
// ONNX models are in the 'onnx' subdirectory
const SMOLVLM_MODEL_BASE_URL = 'https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/resolve/main/onnx';
const TARGET_IMAGE_SIZE = 224; // SmolVLM typically uses 224x224 for vision encoder

/**
 * Tensor shape types for type safety
 * These types ensure correct tensor dimensions and prevent rank mismatches
 */

/**
 * Image tensor shape: [batch, num_images, channels, height, width]
 */
type ImageTensorShape = [number, number, number, number, number];

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
  tokenizer: {
    encode: (text: string) => { ids: number[] };
    decode: (ids: number[]) => string;
  } | null;
  isLoaded: boolean;
}

const smolvlmModel: SmolVLMModel = {
  visionEncoder: null,
  decoder: null,
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
            onLog(`Proxy failed: ${proxy.baseUrl} (Status: ${proxyResponse.status} ${proxyResponse.statusText})`, 'warning');
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
            // Validate that proxy returned binary data, not HTML error page
            // Use clone to peek without consuming original stream
            // Note: If the proxy hangs here, the streaming timeout (30s) will catch it later
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
        // If response is not OK, try next proxy
        lastError = new Error(`Proxy returned status ${proxyResponse.status}: ${proxyResponse.statusText}`);
      } catch (proxyError) {
        // Try next proxy on error
        lastError = proxyError instanceof Error ? proxyError : new Error('Unknown proxy error');
        continue;
      }
    }
    
    // If all proxies failed, try direct fetch as last resort
    if (!response) {
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
 * Load SmolVLM-500M ONNX models and tokenizer with progress tracking
 * @param onProgress - Optional callback to report loading progress (0-100)
 */
export async function loadSmolVLM(onProgress?: ProgressCallback, onLog?: LogCallback): Promise<void> {
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
      }
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
          reportProgress(estimatedProgress, `Loading decoder... ${loadedMB} MB`);
        }
      }
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

    // Load tokenizer (estimated 5% of total, from 95% to 100%)
    reportProgress(95, 'Loading tokenizer...');
    if (onLog) {
      onLog('Checking cache for tokenizer...', 'info');
    }
    const tokenizerUrl = `${SMOLVLM_MODEL_BASE_URL}/tokenizer.json`;
    
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
      onLog('SmolVLM-500M model fully loaded and ready', 'success');
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    if (onLog) {
      onLog(`Model loading failed: ${errorMsg}`, 'error');
    }
    throw new Error(`Failed to load SmolVLM-500M: ${errorMsg}`);
  }
}

/**
 * Format VQA prompt using SmolVLM chat template
 * SmolVLM-500M-Instruct uses a specific format with image and text content
 */
export function formatVQAPrompt(question: string): string {
  // SmolVLM chat template format
  // Format: <image>\nUser: {question}\nAssistant:
  return `<image>\nUser: ${question}\nAssistant:`;
}

/**
 * Tokenize text using the model's tokenizer
 */
export function tokenizeText(text: string): number[] {
  if (!smolvlmModel.tokenizer) {
    throw new Error('Tokenizer not loaded');
  }

  const encoded = smolvlmModel.tokenizer.encode(text);
  return encoded.ids;
}

/**
 * Generate response using SmolVLM-500M
 * @param imageData - Preprocessed image data from WASM (normalized Float32Array)
 * @param question - User's question about the image
 * @returns Generated answer text
 */
export async function generateResponse(
  imageData: Float32Array,
  question: string
): Promise<string> {
  if (!smolvlmModel.isLoaded || !smolvlmModel.visionEncoder || !smolvlmModel.decoder) {
    throw new Error('SmolVLM model not loaded');
  }

  // Format prompt with chat template
  const prompt = formatVQAPrompt(question);
  
  // Tokenize prompt
  const inputIds = tokenizeText(prompt);

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

  // Run vision encoder to get image embeddings
  // ONNX Runtime Web uses Record<string, ort.Tensor> for inputs
  const visionInputs: Record<string, ort.Tensor> = {
    pixel_values: imageTensor,
  };
  
  const visionOutputs = await smolvlmModel.visionEncoder.run(visionInputs);
  
  // Get image embeddings from vision encoder output
  // Output name depends on model - check model outputs
  // Common names: 'last_hidden_state', 'image_embeds', 'pooler_output'
  let imageEmbeddings: ort.Tensor | undefined;
  const outputKeys = Object.keys(visionOutputs);
  const FIRST_OUTPUT_INDEX = 0;
  const MIN_OUTPUTS_REQUIRED = 1;
  if (outputKeys.length >= MIN_OUTPUTS_REQUIRED) {
    const firstOutput = visionOutputs[outputKeys[FIRST_OUTPUT_INDEX]];
    if (firstOutput instanceof ort.Tensor) {
      imageEmbeddings = firstOutput;
    }
  }
  
  if (!imageEmbeddings) {
    throw new Error('Vision encoder did not return embeddings');
  }

  // Prepare decoder inputs
  // Combine image embeddings with text token IDs
  // Note: Actual decoder input format needs verification from model specs
  const INPUT_IDS_BATCH_SIZE = 1;
  const INPUT_IDS_DATA_TYPE = 'int64';
  const inputIdsShape: InputIdsShape = [
    INPUT_IDS_BATCH_SIZE,
    inputIds.length
  ];
  const inputIdsTensor = new ort.Tensor(
    INPUT_IDS_DATA_TYPE,
    new BigInt64Array(inputIds.map(id => BigInt(id))),
    inputIdsShape
  );
  
  const decoderInputs: Record<string, ort.Tensor> = {
    input_ids: inputIdsTensor,
    image_embeds: imageEmbeddings,
  };

  // Run decoder for autoregressive generation
  // This is a simplified version - actual generation loop needs proper implementation
  const decoderOutputs = await smolvlmModel.decoder.run(decoderInputs);
  
  // Extract generated tokens from decoder output
  // Output name depends on model - typically 'logits'
  let logits: ort.Tensor | undefined;
  const decoderOutputKeys = Object.keys(decoderOutputs);
  const FIRST_DECODER_OUTPUT_INDEX = 0;
  const MIN_DECODER_OUTPUTS_REQUIRED = 1;
  if (decoderOutputKeys.length >= MIN_DECODER_OUTPUTS_REQUIRED) {
    const firstOutput = decoderOutputs[decoderOutputKeys[FIRST_DECODER_OUTPUT_INDEX]];
    if (firstOutput instanceof ort.Tensor) {
      logits = firstOutput;
    }
  }
  
  if (!logits) {
    throw new Error('Decoder did not return logits');
  }

  // Get predicted token IDs from logits (argmax)
  // Shape: [batch, sequence, vocab_size]
  // Type guard: ensure logits.data is Float32Array
  const logitsDataRaw = logits.data;
  let logitsData: Float32Array;
  if (logitsDataRaw instanceof Float32Array) {
    logitsData = logitsDataRaw;
  } else {
    // Convert to Float32Array if needed
    // ONNX Runtime Web returns TypedArray, so we can safely convert
    const values: number[] = [];
    // Use Object.getOwnPropertyDescriptor to safely access length
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
  
  // Simplified: take argmax of last sequence position
  // Actual implementation should do proper autoregressive generation
  const VOCAB_SIZE_DIM_INDEX = logitsDims.length - 1;
  const SEQUENCE_LENGTH_DIM_INDEX = logitsDims.length - 2;
  const LAST_POSITION_OFFSET = 1;
  const ARGMAX_START_INDEX = 0;
  const ARGMAX_LOOP_START = 1;
  
  const vocabSize = logitsDims[VOCAB_SIZE_DIM_INDEX];
  const sequenceLength = logitsDims[SEQUENCE_LENGTH_DIM_INDEX];
  const lastPositionIdx = (sequenceLength - LAST_POSITION_OFFSET) * vocabSize;
  
  let maxIdx = ARGMAX_START_INDEX;
  let maxVal = logitsData[lastPositionIdx];
  for (let i = ARGMAX_LOOP_START; i < vocabSize; i++) {
    const val = logitsData[lastPositionIdx + i];
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  
  // For now, return a simple response
  // TODO: Implement proper autoregressive generation loop
  if (!smolvlmModel.tokenizer) {
    throw new Error('Tokenizer not loaded');
  }
  
  // Decode the predicted token
  const generatedText = smolvlmModel.tokenizer.decode([maxIdx]);
  
  return generatedText.trim();
}

/**
 * Check if SmolVLM model is loaded
 */
export function isSmolVLMLoaded(): boolean {
  return smolvlmModel.isLoaded;
}

