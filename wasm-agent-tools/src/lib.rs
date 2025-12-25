use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Calculate a mathematical expression
/// Returns the result as a string
/// Supports basic arithmetic: +, -, *, /, parentheses
#[wasm_bindgen]
pub fn calculate(expression: &str) -> Result<String, JsValue> {
    // Simple expression evaluator for basic arithmetic
    // This is a simplified version - in production, use a proper parser
    let trimmed = expression.trim();
    
    // Validate expression contains only allowed characters
    if trimmed.is_empty() {
        return Err(JsValue::from_str("Empty expression"));
    }
    
    // Check for dangerous patterns (no function calls, no eval-like operations)
    if trimmed.contains("eval") || trimmed.contains("exec") || trimmed.contains("import") {
        return Err(JsValue::from_str("Invalid expression"));
    }
    
    // Simple arithmetic evaluation
    // This is a basic implementation - for production, use a proper math parser
    // For now, we'll use a simple approach that handles basic operations
    let result = evaluate_expression(trimmed)?;
    Ok(result.to_string())
}

/// Evaluate a simple arithmetic expression
/// Supports: +, -, *, /, parentheses, numbers
fn evaluate_expression(expr: &str) -> Result<f64, JsValue> {
    // Remove whitespace
    let expr = expr.replace(' ', "");
    
    // Basic validation
    if expr.is_empty() {
        return Err(JsValue::from_str("Empty expression"));
    }
    
    // Simple recursive descent parser for basic arithmetic
    let (result, pos) = parse_expression(&expr, 0).map_err(|e| JsValue::from_str(&e))?;
    
    // Check if entire expression was parsed
    if pos < expr.len() {
        return Err(JsValue::from_str("Invalid expression"));
    }
    
    Ok(result)
}

/// Parse and evaluate expression
fn parse_expression(expr: &str, start: usize) -> Result<(f64, usize), String> {
    let mut pos = start;
    let mut result = parse_term(expr, &mut pos)?;
    
    while pos < expr.len() {
        let ch = expr.chars().nth(pos).unwrap();
        match ch {
            '+' => {
                pos += 1;
                let term = parse_term(expr, &mut pos)?;
                result += term;
            }
            '-' => {
                pos += 1;
                let term = parse_term(expr, &mut pos)?;
                result -= term;
            }
            _ => break,
        }
    }
    
    Ok((result, pos))
}

/// Parse a term (multiplication/division)
fn parse_term(expr: &str, pos: &mut usize) -> Result<f64, String> {
    let mut result = parse_factor(expr, pos)?;
    
    while *pos < expr.len() {
        let ch = expr.chars().nth(*pos).unwrap();
        match ch {
            '*' => {
                *pos += 1;
                let factor = parse_factor(expr, pos)?;
                result *= factor;
            }
            '/' => {
                *pos += 1;
                let factor = parse_factor(expr, pos)?;
                if factor == 0.0 {
                    return Err("Division by zero".to_string());
                }
                result /= factor;
            }
            _ => break,
        }
    }
    
    Ok(result)
}

/// Parse a factor (number or parenthesized expression)
fn parse_factor(expr: &str, pos: &mut usize) -> Result<f64, String> {
    if *pos >= expr.len() {
        return Err("Unexpected end of expression".to_string());
    }
    
    let ch = expr.chars().nth(*pos).unwrap();
    
    if ch == '(' {
        *pos += 1;
        let (result, new_pos) = parse_expression(expr, *pos)?;
        *pos = new_pos;
        if *pos >= expr.len() || expr.chars().nth(*pos).unwrap() != ')' {
            return Err("Missing closing parenthesis".to_string());
        }
        *pos += 1;
        Ok(result)
    } else if ch.is_ascii_digit() || ch == '.' || ch == '-' {
        parse_number(expr, pos)
    } else {
        Err(format!("Unexpected character: {}", ch))
    }
}

/// Parse a number
fn parse_number(expr: &str, pos: &mut usize) -> Result<f64, String> {
    let start = *pos;
    let mut has_dot = false;
    
    // Handle negative sign
    if *pos < expr.len() && expr.chars().nth(*pos).unwrap() == '-' {
        *pos += 1;
    }
    
    // Parse digits
    while *pos < expr.len() {
        let ch = expr.chars().nth(*pos).unwrap();
        if ch.is_ascii_digit() {
            *pos += 1;
        } else if ch == '.' && !has_dot {
            has_dot = true;
            *pos += 1;
        } else {
            break;
        }
    }
    
    if *pos == start || (*pos == start + 1 && expr.chars().nth(start).unwrap() == '-') {
        return Err("Invalid number".to_string());
    }
    
    let num_str = &expr[start..*pos];
    num_str.parse::<f64>().map_err(|_| format!("Invalid number: {}", num_str))
}

/// Process text with various operations
/// operation: "uppercase", "lowercase", "reverse", "length", "word_count"
/// Returns processed text or operation result as string
#[wasm_bindgen]
pub fn process_text(text: &str, operation: &str) -> Result<String, JsValue> {
    match operation {
        "uppercase" => Ok(text.to_uppercase()),
        "lowercase" => Ok(text.to_lowercase()),
        "reverse" => Ok(text.chars().rev().collect()),
        "length" => Ok(text.len().to_string()),
        "word_count" => {
            let count = if text.trim().is_empty() {
                0
            } else {
                text.split_whitespace().count()
            };
            Ok(count.to_string())
        }
        _ => Err(JsValue::from_str(&format!("Unknown operation: {}", operation))),
    }
}

/// Get statistics from data
/// Returns statistics as a JSON string
#[wasm_bindgen]
pub fn get_stats(data: &[u8]) -> Result<String, JsValue> {
    if data.is_empty() {
        return Ok(r#"{"count":0,"min":0,"max":0,"sum":0,"average":0}"#.to_string());
    }
    
    let count = data.len();
    let min = *data.iter().min().unwrap_or(&0);
    let max = *data.iter().max().unwrap_or(&0);
    let sum: u64 = data.iter().map(|&x| x as u64).sum();
    let average = sum as f64 / count as f64;
    
    // Build JSON string manually (no serde dependency to keep WASM small)
    let json = format!(
        r#"{{"count":{},"min":{},"max":{},"sum":{},"average":{:.2}}}"#,
        count, min, max, sum, average
    );
    
    Ok(json)
}

