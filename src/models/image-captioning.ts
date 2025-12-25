import { pipeline, type Pipeline, env } from '@xenova/transformers';

// Model configuration
// Transformers.js supports specific model architectures
// Try models that are known to work with Transformers.js
// Options: 
// - 'Xenova/vit-gpt2-image-captioning' (ViT-GPT2, supported)
// - 'Xenova/git-base' (GIT model, supported)
// - 'Xenova/blip-image-captioning-base' (not supported by Transformers.js)
const MODEL_ID = 'Xenova/vit-gpt2-image-captioning';

// CORS proxy services for Hugging Face model loading
const CORS_PROXY_SERVICES = [
  'https://api.allorigins.win/raw?url=',
  'https://corsproxy.io/?',
  'https://api.codetabs.com/v1/proxy?quest=',
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
    !url.includes('api.codetabs.com')
  );
}

/**
 * Custom fetch function with CORS proxy support
 */
async function customFetch(input: RequestInfo | URL, init?: RequestInit, onLog?: LogCallback): Promise<Response> {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
  
  // If URL doesn't need proxying, use normal fetch
  if (!needsProxy(url)) {
    return fetch(input, init);
  }
  
  if (onLog) {
    onLog(`Custom fetch: Attempting to fetch via CORS proxy: ${url}`, 'info');
  }
  
  // Try each CORS proxy in order
  for (const proxyBase of CORS_PROXY_SERVICES) {
    try {
      const proxyUrl = proxyBase + encodeURIComponent(url);
      if (onLog) {
        onLog(`Trying proxy: ${proxyBase}`, 'info');
      }
      
      const response = await fetch(proxyUrl, {
        ...init,
        redirect: 'follow',
      });
      
      // Skip proxies that return error status codes
      if (response.status >= 400 && response.status < 600) {
        if (onLog) {
          onLog(`Proxy ${proxyBase} returned error: ${response.status} ${response.statusText}`, 'warning');
        }
        continue;
      }
      
      // If response looks good, return it
      if (response.ok) {
        if (onLog) {
          onLog(`Successfully fetched via proxy: ${proxyBase}`, 'success');
        }
        return response;
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      if (onLog) {
        onLog(`Proxy ${proxyBase} failed: ${errorMsg}`, 'warning');
      }
      // Try next proxy
      continue;
    }
  }
  
  if (onLog) {
    onLog('All CORS proxies failed, trying direct fetch as last resort', 'warning');
  }
  
  // If all proxies fail, try direct fetch as last resort
  return fetch(input, init);
}

// Configure Transformers.js environment
// Disable local model loading to force remote fetching from Hugging Face
env.allowLocalModels = false;

// Enable caching - Transformers.js uses IndexedDB by default
// The cache is automatically used when loading models

// Override the fetch function to use CORS proxies
// Note: env.fetch may not be in TypeScript types but is available at runtime
// We'll set this up when loadImageCaptioningModel is called so we have access to the log callback
const setupCustomFetch = (onLog?: LogCallback): void => {
  // Use proper type narrowing instead of type assertion
  if (typeof env === 'object' && env !== null) {
    const envRecord: Record<string, unknown> = env;
    envRecord.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      return customFetch(input, init, onLog);
    };
  }
};

// Pipeline instance
let imageToTextPipeline: Pipeline | null = null;

// Loading state
let isLoading = false;
let isLoaded = false;

// Progress callback type
export type ProgressCallback = (progress: number) => void;
export type LogCallback = (message: string, type?: 'info' | 'success' | 'warning' | 'error') => void;

/**
 * Load the ViT-GPT2 image-to-text model
 */
export async function loadImageCaptioningModel(
  onProgress?: ProgressCallback,
  onLog?: LogCallback
): Promise<void> {
  if (isLoaded && imageToTextPipeline) {
    if (onLog) {
      onLog('Image captioning model already loaded', 'info');
    }
    return;
  }

  if (isLoading) {
    if (onLog) {
      onLog('Image captioning model is already loading...', 'info');
    }
    return;
  }

  isLoading = true;

  try {
    // Set up custom fetch with logging before loading model
    setupCustomFetch(onLog);
    
    if (onLog) {
      onLog(`Loading image captioning model: ${MODEL_ID}...`, 'info');
      onLog('Checking for cached model...', 'info');
    }

    // Check if model is already cached in IndexedDB
    // Transformers.js stores models in IndexedDB with keys like 'models--Xenova--vit-gpt2-image-captioning'
    let fromCache = false;
    try {
      if ('indexedDB' in window) {
        const dbName = 'transformers-cache';
        const request = indexedDB.open(dbName);
        await new Promise<void>((resolve) => {
          request.onsuccess = () => {
            const db = request.result;
            if (db.objectStoreNames.contains('models')) {
              const transaction = db.transaction(['models'], 'readonly');
              const store = transaction.objectStore('models');
              // Transformers.js uses model ID as key
              const modelKey = `models--${MODEL_ID.replace('/', '--')}`;
              const getRequest = store.get(modelKey);
              getRequest.onsuccess = () => {
                if (getRequest.result) {
                  fromCache = true;
                  if (onLog) {
                    onLog('Model found in cache!', 'success');
                  }
                } else {
                  if (onLog) {
                    onLog('Model not in cache, will download...', 'info');
                  }
                }
                db.close();
                resolve();
              };
              getRequest.onerror = () => {
                db.close();
                resolve(); // Continue even if check fails
              };
            } else {
              db.close();
              resolve();
            }
          };
          request.onerror = () => {
            resolve(); // Continue even if IndexedDB check fails
          };
        });
      }
    } catch {
      // IndexedDB check failed, continue with normal loading
      if (onLog) {
        onLog('Could not check cache, proceeding with load...', 'info');
      }
    }

    if (onProgress) {
      onProgress(fromCache ? 100 : 0);
    }

    // Create the image-to-text pipeline
    // Transformers.js will automatically download and cache the model
    // The custom fetch function is set via env.fetch override above
    let downloadStarted = false;
    imageToTextPipeline = await pipeline('image-to-text', MODEL_ID, {
      progress_callback: (progress: { progress: number; loaded: number; total: number }) => {
        // If we detected cache but progress callback fires, it might be loading additional files
        if (!downloadStarted && progress.total > 0 && progress.loaded > 0) {
          downloadStarted = true;
          if (onLog && !fromCache) {
            onLog('Starting model download...', 'info');
          } else if (onLog) {
            onLog('Loading additional model files...', 'info');
          }
        }
        
        if (onProgress && progress.total > 0) {
          const percent = Math.round((progress.loaded / progress.total) * 100);
          onProgress(percent);
        } else if (onProgress && !fromCache) {
          // If total is unknown, show indeterminate progress
          onProgress(50);
        }

        if (onLog && progress.total > 0) {
          const loadedMB = (progress.loaded / (1024 * 1024)).toFixed(2);
          const totalMB = (progress.total / (1024 * 1024)).toFixed(2);
          if (fromCache && !downloadStarted) {
            onLog(`Loading from cache: ${loadedMB} MB / ${totalMB} MB (${Math.round((progress.loaded / progress.total) * 100)}%)`, 'info');
          } else {
            onLog(`Downloading model: ${loadedMB} MB / ${totalMB} MB (${Math.round((progress.loaded / progress.total) * 100)}%)`, 'info');
          }
        }
      },
    });

    if (onProgress) {
      onProgress(100);
    }

    if (onLog) {
      onLog('Image captioning model loaded successfully', 'success');
    }

    isLoaded = true;
  } catch (error) {
    isLoading = false;
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    // Provide more detailed error information
    let detailedError = errorMessage;
    if (errorMessage.includes('Unexpected token') || errorMessage.includes('DOCTYPE')) {
      detailedError = `${errorMessage}. This usually means the model files could not be downloaded from Hugging Face (CORS issue or model not found). Try refreshing the page or check your network connection.`;
    }
    
    if (onLog) {
      onLog(`Failed to load image captioning model: ${detailedError}`, 'error');
      onLog(`Model ID attempted: ${MODEL_ID}`, 'info');
      onLog('Note: Transformers.js models are downloaded from Hugging Face. If this fails, it may be a CORS or network issue.', 'info');
    }
    throw new Error(`Failed to load image captioning model: ${detailedError}`);
  } finally {
    isLoading = false;
  }
}

/**
 * Generate image caption using ViT-GPT2
 */
export async function generateCaption(
  imageData: ImageData | HTMLImageElement | HTMLCanvasElement | string,
  onLog?: LogCallback
): Promise<string> {
  if (!imageToTextPipeline) {
    throw new Error('Image captioning model not loaded. Call loadImageCaptioningModel() first.');
  }

  try {
    if (onLog) {
      onLog('Generating image caption...', 'info');
    }

    // Run inference
    // Pipeline returns unknown, so we need to narrow it properly
    const result: unknown = await imageToTextPipeline(imageData);

    // Extract the generated text
    // The result is typically an array with objects containing 'generated_text'
    // Use proper type narrowing without any types
    let caption = '';
    
    if (Array.isArray(result) && result.length > 0) {
      const firstResult: unknown = result[0];
      if (typeof firstResult === 'object' && firstResult !== null) {
        // Access properties directly using 'in' operator for type narrowing
        if ('generated_text' in firstResult) {
          const generatedText: unknown = firstResult.generated_text;
          if (typeof generatedText === 'string') {
            caption = generatedText;
          }
        }
        if (caption === '' && 'text' in firstResult) {
          const text: unknown = firstResult.text;
          if (typeof text === 'string') {
            caption = text;
          }
        }
      }
    } else if (typeof result === 'object' && result !== null) {
      // Access properties directly using 'in' operator for type narrowing
      if ('generated_text' in result) {
        const generatedText: unknown = result.generated_text;
        if (typeof generatedText === 'string') {
          caption = generatedText;
        }
      }
      if (caption === '' && 'text' in result) {
        const text: unknown = result.text;
        if (typeof text === 'string') {
          caption = text;
        }
      }
    }

    if (onLog) {
      onLog(`Caption generated: ${caption.substring(0, 100)}${caption.length > 100 ? '...' : ''}`, 'success');
    }

    return caption || 'No caption generated';
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    if (onLog) {
      onLog(`Image captioning error: ${errorMessage}`, 'error');
    }
    throw new Error(`Image captioning error: ${errorMessage}`);
  }
}


/**
 * Check if image captioning model is loaded
 */
export function isImageCaptioningModelLoaded(): boolean {
  return isLoaded && imageToTextPipeline !== null;
}

/**
 * Get the model ID being used
 */
export function getModelId(): string {
  return MODEL_ID;
}

