use wasm_bindgen::prelude::*;
use image::{io::Reader as ImageReader, ImageFormat, GenericImageView};
use std::io::Cursor;
use std::sync::{LazyLock, Mutex};

// State management pattern similar to wasm-astar
// Learned about this pattern from rocket_wasm on github
// https://github.com/aochagavia/rocket_wasm/blob/d0ca51beb9c7c351a1f0266206edfd553bf078d3/src/lib.rs
struct PreprocessState {
    contrast: f32,
    cinematic: f32,
}

impl PreprocessState {
    fn new() -> Self {
        PreprocessState {
            contrast: 0.0,
            cinematic: 0.0,
        }
    }
    
    fn set_contrast(&mut self, contrast: f32) {
        self.contrast = contrast;
    }
    
    fn set_cinematic(&mut self, cinematic: f32) {
        self.cinematic = cinematic;
    }
    
    fn get_contrast(&self) -> f32 {
        self.contrast
    }
    
    fn get_cinematic(&self) -> f32 {
        self.cinematic
    }
}

static PREPROCESS_STATE: LazyLock<Mutex<PreprocessState>> = LazyLock::new(|| Mutex::new(PreprocessState::new()));

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Preprocess image data by resizing to target dimensions using high-quality Lanczos3 filtering
/// Returns preprocessed image data as RGBA bytes
/// This is a building block for ML/AI preprocessing pipelines
/// Note: source_width and source_height are kept for API compatibility but dimensions are determined from decoded image
#[wasm_bindgen]
pub fn preprocess_image(
    image_data: &[u8],
    _source_width: u32,
    _source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, JsValue> {
    // Copy the image data into a Vec to ensure proper memory management
    // This prevents issues with WASM memory deallocation
    let image_bytes = image_data.to_vec();
    
    // Decode image from bytes (supports PNG and JPEG)
    // Try PNG first, then JPEG
    let img = ImageReader::with_format(Cursor::new(&image_bytes), ImageFormat::Png)
        .decode()
        .or_else(|_| {
            ImageReader::with_format(Cursor::new(&image_bytes), ImageFormat::Jpeg)
                .decode()
        })
        .map_err(|e| JsValue::from_str(&format!("Failed to decode image: {}", e)))?;

    // Resize using Lanczos3 filter for high-quality resizing
    // Lanczos3 provides excellent quality for ML model preprocessing
    let resized_img = img.resize_exact(target_width, target_height, image::imageops::FilterType::Lanczos3);

    // Convert to RGBA format
    let rgba_img = resized_img.to_rgba8();
    
    // Return as Vec<u8> (RGBA bytes)
    Ok(rgba_img.into_raw())
}

/// Preprocess image data by center cropping to square then resizing to target dimensions
/// Preserves aspect ratio by cropping instead of distorting
/// Returns preprocessed image data as RGBA bytes
/// This is preferred for ML/AI preprocessing when models require square inputs
#[wasm_bindgen]
pub fn preprocess_image_crop(
    image_data: &[u8],
    _source_width: u32,
    _source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, JsValue> {
    // Copy the image data into a Vec to ensure proper memory management
    // This prevents issues with WASM memory deallocation
    let image_bytes = image_data.to_vec();
    
    // Decode image from bytes (supports PNG and JPEG)
    // Try PNG first, then JPEG
    let img = ImageReader::with_format(Cursor::new(&image_bytes), ImageFormat::Png)
        .decode()
        .or_else(|_| {
            ImageReader::with_format(Cursor::new(&image_bytes), ImageFormat::Jpeg)
                .decode()
        })
        .map_err(|e| JsValue::from_str(&format!("Failed to decode image: {}", e)))?;

    let (img_width, img_height) = img.dimensions();
    
    // Calculate center crop to square (use smaller dimension)
    let crop_size = img_width.min(img_height);
    let crop_x = (img_width - crop_size) / 2;
    let crop_y = (img_height - crop_size) / 2;
    
    // Crop to square
    let cropped_img = img.crop_imm(crop_x, crop_y, crop_size, crop_size);
    
    // Resize cropped square to target dimensions using Lanczos3
    let resized_img = cropped_img.resize_exact(target_width, target_height, image::imageops::FilterType::Lanczos3);

    // Convert to RGBA format
    let rgba_img = resized_img.to_rgba8();
    
    // Return as Vec<u8> (RGBA bytes)
    Ok(rgba_img.into_raw())
}

/// Preprocess image data specifically for SmolVLM-256M model
/// Performs: decode, center crop, resize, RGB conversion, normalization
/// Returns normalized Float32Array (shape: [height * width * 3]) for ONNX Runtime
/// Normalization: pixel values normalized to [0.0, 1.0] range
#[wasm_bindgen]
pub fn preprocess_image_for_smolvlm_256m(
    image_data: &[u8],
    _source_width: u32,
    _source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<f32>, JsValue> {
    // Copy the image data into a Vec to ensure proper memory management
    // This prevents issues with WASM memory deallocation
    let image_bytes = image_data.to_vec();
    
    // Decode image from bytes (supports PNG and JPEG)
    // Try PNG first, then JPEG
    let img = ImageReader::with_format(Cursor::new(&image_bytes), ImageFormat::Png)
        .decode()
        .or_else(|_| {
            ImageReader::with_format(Cursor::new(&image_bytes), ImageFormat::Jpeg)
                .decode()
        })
        .map_err(|e| JsValue::from_str(&format!("Failed to decode image: {}", e)))?;

    let (img_width, img_height) = img.dimensions();
    
    // Calculate center crop to square (use smaller dimension)
    let crop_size = img_width.min(img_height);
    let crop_x = (img_width - crop_size) / 2;
    let crop_y = (img_height - crop_size) / 2;
    
    // Crop to square
    let cropped_img = img.crop_imm(crop_x, crop_y, crop_size, crop_size);
    
    // Resize cropped square to target dimensions using Lanczos3
    let resized_img = cropped_img.resize_exact(target_width, target_height, image::imageops::FilterType::Lanczos3);

    // Convert to RGB format (remove alpha channel)
    let rgb_img = resized_img.to_rgb8();
    
    // Normalize pixel values to [0.0, 1.0] range and convert to Float32Array
    // Output format: [R, G, B, R, G, B, ...] flattened (height * width * 3)
    let mut normalized_data = Vec::with_capacity((target_width * target_height * 3) as usize);
    
    for pixel in rgb_img.pixels() {
        // Normalize each channel: pixel_value / 255.0
        normalized_data.push(pixel[0] as f32 / 255.0); // R
        normalized_data.push(pixel[1] as f32 / 255.0); // G
        normalized_data.push(pixel[2] as f32 / 255.0); // B
    }
    
    Ok(normalized_data)
}

/// Apply contrast enhancement to RGBA image data
/// contrast: -100.0 to 100.0 (0.0 = no change, positive = increase, negative = decrease)
/// Returns processed image data as RGBA bytes
#[wasm_bindgen]
pub fn apply_contrast(
    image_data: &[u8],
    width: u32,
    height: u32,
    contrast: f32,
) -> Result<Vec<u8>, JsValue> {
    if image_data.len() != (width * height * 4) as usize {
        return Err(JsValue::from_str("Image data size mismatch"));
    }
    
    let mut result = Vec::with_capacity(image_data.len());
    
    // Convert contrast from -100.0..100.0 to multiplier
    // Formula: factor = (100.0 + contrast) / 100.0
    // Then apply: new_value = ((old_value - 128) * factor) + 128
    let factor = (100.0 + contrast) / 100.0;
    
    for chunk in image_data.chunks_exact(4) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let a = chunk[3];
        
        // Apply contrast to RGB channels (not alpha)
        let new_r = ((r - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
        let new_g = ((g - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
        let new_b = ((b - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
        
        result.push(new_r);
        result.push(new_g);
        result.push(new_b);
        result.push(a);
    }
    
    Ok(result)
}

/// Apply cinematic filter to RGBA image data
/// intensity: 0.0 to 1.0 (0.0 = no effect, 1.0 = full cinematic effect)
/// Cinematic filter: desaturates slightly, adds blue/teal tint, increases contrast
/// Returns processed image data as RGBA bytes
#[wasm_bindgen]
pub fn apply_cinematic_filter(
    image_data: &[u8],
    width: u32,
    height: u32,
    intensity: f32,
) -> Result<Vec<u8>, JsValue> {
    if image_data.len() != (width * height * 4) as usize {
        return Err(JsValue::from_str("Image data size mismatch"));
    }
    
    let intensity = intensity.clamp(0.0, 1.0);
    let mut result = Vec::with_capacity(image_data.len());
    
    for chunk in image_data.chunks_exact(4) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let a = chunk[3];
        
        // Calculate luminance for desaturation
        let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        
        // Blend between original and desaturated based on intensity
        let desat_r = luminance;
        let desat_g = luminance;
        let desat_b = luminance;
        
        let new_r = (r * (1.0 - intensity) + desat_r * intensity).clamp(0.0, 255.0) as u8;
        let new_g = (g * (1.0 - intensity) + desat_g * intensity).clamp(0.0, 255.0) as u8;
        let mut new_b = (b * (1.0 - intensity) + desat_b * intensity).clamp(0.0, 255.0) as u8;
        
        // Add blue/teal tint (increase blue channel slightly)
        new_b = ((new_b as f32) * (1.0 + intensity * 0.1)).clamp(0.0, 255.0) as u8;
        
        // Slight contrast boost for cinematic look
        let contrast_boost = 1.0 + (intensity * 0.15);
        let final_r = (((new_r as f32 - 128.0) * contrast_boost) + 128.0).clamp(0.0, 255.0) as u8;
        let final_g = (((new_g as f32 - 128.0) * contrast_boost) + 128.0).clamp(0.0, 255.0) as u8;
        let final_b = (((new_b as f32 - 128.0) * contrast_boost) + 128.0).clamp(0.0, 255.0) as u8;
        
        result.push(final_r);
        result.push(final_g);
        result.push(final_b);
        result.push(a);
    }
    
    Ok(result)
}

/// Get preprocessing statistics
#[wasm_bindgen]
pub fn get_preprocess_stats(
    original_size: u32,
    target_size: u32,
) -> PreprocessStats {
    PreprocessStats {
        original_size,
        target_size,
        scale_factor: target_size as f64 / original_size as f64,
    }
}

#[wasm_bindgen]
pub struct PreprocessStats {
    pub original_size: u32,
    pub target_size: u32,
    pub scale_factor: f64,
}

/// Set contrast value in WASM state
/// Similar pattern to mouse_move in wasm-astar
#[wasm_bindgen]
pub fn set_contrast(contrast: f32) {
    let state = &mut PREPROCESS_STATE.lock().unwrap();
    state.set_contrast(contrast);
}

/// Set cinematic filter intensity in WASM state
/// Similar pattern to mouse_move in wasm-astar
#[wasm_bindgen]
pub fn set_cinematic(intensity: f32) {
    let state = &mut PREPROCESS_STATE.lock().unwrap();
    state.set_cinematic(intensity);
}

/// Get current contrast value from WASM state
#[wasm_bindgen]
pub fn get_contrast() -> f32 {
    let state = PREPROCESS_STATE.lock().unwrap();
    state.get_contrast()
}

/// Get current cinematic intensity from WASM state
#[wasm_bindgen]
pub fn get_cinematic() -> f32 {
    let state = PREPROCESS_STATE.lock().unwrap();
    state.get_cinematic()
}

