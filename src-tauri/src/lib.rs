use std::sync::Mutex;
use tauri::{Manager, State};

struct TranslatorState(Mutex<translator::translator::Translator>);

#[tauri::command]
fn read_selection() -> Result<String, String> {
    translator::read_selection()
}

#[tauri::command]
fn translate(text: String, state: State<'_, TranslatorState>) -> Result<String, String> {
    let mut translator = state.0.lock().map_err(|e| e.to_string())?;
    translator.translate(&text)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let model_path = app
                .path()
                .resource_dir()
                .expect("failed to resolve resource dir")
                .join("models/model-fp32.onnx");
            let tokenizer_path = app
                .path()
                .resource_dir()
                .expect("failed to resolve resource dir")
                .join("models/tokenizer.json");

            // Fallback: try current working directory (dev mode)
            let (model_path, tokenizer_path) = if model_path.exists() {
                (model_path, tokenizer_path)
            } else {
                (
                    std::path::PathBuf::from("models/model-fp32.onnx"),
                    std::path::PathBuf::from("models/tokenizer.json"),
                )
            };

            let model_str = model_path.to_str().unwrap_or("models/model-fp32.onnx");
            let tok_str = tokenizer_path
                .to_str()
                .unwrap_or("models/tokenizer.json");

            match translator::translator::Translator::load_model(model_str, tok_str) {
                Ok(translator) => {
                    log::info!("ONNX model loaded from: {}", model_str);
                    app.manage(TranslatorState(Mutex::new(translator)));
                }
                Err(e) => {
                    log::error!("Failed to load model: {}", e);
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![read_selection, translate])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
