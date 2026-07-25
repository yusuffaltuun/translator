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
    let output = Command::new("wl-paste")
        .args(["--primary"])
        .output()
        .map_err(|e| format!("failed to run wl-paste: {}", e))?;

    if !output.status.success() {
        return Err(format!("wl-paste exited with: {}", output.status));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("invalid UTF-8: {}", e))
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

    let model_path = "models/model-fp32.onnx";
    let tokenizer_path = "models/tokenizer.json";

    match Translator::load_model(model_path, tokenizer_path) {
        Ok(mut translator) => {
            println!("Model loaded successfully from: {}", model_path);

            let sample = "Hello, how are you?";
            println!("Translating: \"{}\"", sample);
            match translator.translate(sample) {
                Ok(translation) => println!("Translation: \"{}\"", translation),
                Err(e) => eprintln!("Translation error: {}", e),
            }
        }
        Err(e) => eprintln!("Failed to load model: {}", e),
    }
}
