pub mod translator;

use std::env;
use std::process::Command;

use translator::Translator;

#[derive(Debug, PartialEq)]
pub enum DisplayServer {
    Wayland,
    X11,
    Unknown,
}

pub fn detect_display_server() -> DisplayServer {
    match env::var("XDG_SESSION_TYPE").unwrap_or_default().as_str() {
        "wayland" => DisplayServer::Wayland,
        "x11" => DisplayServer::X11,
        _ => {
            if env::var("WAYLAND_DISPLAY").is_ok() {
                DisplayServer::Wayland
            } else if env::var("DISPLAY").is_ok() {
                DisplayServer::X11
            } else {
                DisplayServer::Unknown
            }
        }
    }
}

pub fn read_x11_selection() -> Result<String, String> {
    let output = Command::new("xclip")
        .args(["-selection", "primary", "-o"])
        .output()
        .map_err(|e| format!("failed to run xclip: {}", e))?;

    if !output.status.success() {
        return Err(format!("xclip exited with: {}", output.status));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("invalid UTF-8: {}", e))
}

pub fn read_wayland_selection() -> Result<String, String> {
    // Try --primary first, fall back to -p (some compositors differ)
    let output = Command::new("wl-paste")
        .args(["--primary"])
        .output()
        .or_else(|_| Command::new("wl-paste").args(["-p"]).output())
        .map_err(|e| format!("wl-paste not found (install wl-clipboard): {}", e))?;

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        if code == 1 && (stderr.contains("no selection") || stderr.contains("empty")) {
            return Err("No primary selection found. Select some text first.".into());
        }
        return Err(format!("wl-paste exited {}: {}", code, stderr.trim()));
    }

    let text = String::from_utf8(output.stdout)
        .map_err(|e| format!("invalid UTF-8 in clipboard: {}", e))?;

    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        return Err("Primary selection is empty. Select some text first.".into());
    }

    Ok(trimmed)
}

pub fn read_selection() -> Result<String, String> {
    match detect_display_server() {
        DisplayServer::X11 => read_x11_selection(),
        DisplayServer::Wayland => read_wayland_selection(),
        DisplayServer::Unknown => Err("Unknown display server".into()),
    }
}

pub fn main() {
    let server = detect_display_server();
    println!("Display server detected: {:?}", server);

    match read_selection() {
        Ok(text) => println!("Selection: {}", text),
        Err(e) => eprintln!("Error: {}", e),
    }

    // Prefer INT8-quantized models; fall back to FP32.
    let (encoder_path, decoder_path) = if std::path::Path::new("models/encoder_model_int8.onnx").exists()
        && std::path::Path::new("models/decoder_model_int8.onnx").exists()
    {
        ("models/encoder_model_int8.onnx", "models/decoder_model_int8.onnx")
    } else {
        ("models/encoder_model.onnx", "models/decoder_model.onnx")
    };
    let tokenizer_path = "models/tokenizer.json";

    match Translator::load_model(encoder_path, decoder_path, tokenizer_path) {
        Ok(mut translator) => {
            println!("Model loaded successfully from: {} + {}", encoder_path, decoder_path);

            let args: Vec<String> = std::env::args().skip(1).collect();
            let sample = if !args.is_empty() {
                args.join(" ")
            } else {
                "Hello, how are you?".to_string()
            };
            println!("Translating: \"{}\"", sample);
            match translator.translate(&sample) {
                Ok(translation) => println!("Translation: \"{}\"", translation),
                Err(e) => eprintln!("Translation error: {}", e),
            }
        }
        Err(e) => eprintln!("Failed to load model: {}", e),
    }
}
