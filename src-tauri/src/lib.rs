use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::{Router, extract::State as AxumState, routing::get};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri::WebviewWindow;

use translator::translator::DefaultTranslator;

struct TranslatorState(Arc<Mutex<DefaultTranslator>>);

struct AppState {
    translator: Arc<Mutex<DefaultTranslator>>,
    app: AppHandle,
}

#[tauri::command]
async fn read_selection() -> Result<String, String> {
    tokio::task::spawn_blocking(translator::read_selection)
        .await
        .map_err(|e| format!("task join error: {}", e))?
}

#[tauri::command]
async fn translate(text: String, state: State<'_, TranslatorState>) -> Result<String, String> {
    let translator = state.0.clone();
    tokio::task::spawn_blocking(move || -> Result<String, String> {
        let mut t = translator.lock().map_err(|e| e.to_string())?;
        t.translate(&text).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("task join error: {}", e))?
}

async fn selection_translate(
    AxumState(state): AxumState<Arc<AppState>>,
) -> axum::Json<serde_json::Value> {
    let window: Option<WebviewWindow> = state.app.get_webview_window("main");

    if let Some(w) = &window {
        let _ = w.show();
        let _ = w.set_focus();
    }
    log::info!("overlay shown via HTTP, reading selection…");

    let translator = state.translator.clone();
    let app = state.app.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<(String, String), String> {
        let text = translator::read_selection()?;
        if text.trim().is_empty() {
            return Err("No primary selection found. Select some text first.".into());
        }
        let mut t = translator.lock().map_err(|e| e.to_string())?;
        let tr = t.translate(&text).map_err(|e| e.to_string())?;
        Ok((text, tr))
    })
    .await;

    let payload = match result {
        Ok(Ok((source, translation))) => serde_json::json!({
            "source": source,
            "translation": translation
        }),
        Ok(Err(e)) => serde_json::json!({
            "source": "",
            "error": e
        }),
        Err(e) => serde_json::json!({
            "source": "",
            "error": format!("task failed: {}", e)
        }),
    };

    let _ = app.emit("translate-result", &payload);
    axum::Json(payload)
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

            // ── Load ONNX model ──
            // Helsinki-NLP/opus-mt-tc-big-en-tr (Tier 2 upgrade).
            // Uses separate encoder + decoder ONNX models.
            // Prefer INT8-quantized models; fall back to FP32.
            let resource_dir = app
                .path()
                .resource_dir()
                .expect("failed to resolve resource dir");

            let (encoder_path, decoder_path, tokenizer_path) = {
                let int8_enc = resource_dir.join("models/encoder_model_int8.onnx");
                let int8_dec = resource_dir.join("models/decoder_model_int8.onnx");
                let fp32_enc = resource_dir.join("models/encoder_model.onnx");
                let fp32_dec = resource_dir.join("models/decoder_model.onnx");
                let tok = resource_dir.join("models/tokenizer.json");

                if int8_enc.exists() && int8_dec.exists() {
                    (int8_enc, int8_dec, tok)
                } else if fp32_enc.exists() && fp32_dec.exists() {
                    (fp32_enc, fp32_dec, tok)
                } else {
                    // Dev / cargo run fallback — look in crate root
                    let dev_int8_enc = std::path::PathBuf::from("models/encoder_model_int8.onnx");
                    let dev_int8_dec = std::path::PathBuf::from("models/decoder_model_int8.onnx");
                    let dev_fp32_enc = std::path::PathBuf::from("models/encoder_model.onnx");
                    let dev_fp32_dec = std::path::PathBuf::from("models/decoder_model.onnx");
                    let dev_tok = std::path::PathBuf::from("models/tokenizer.json");
                    if dev_int8_enc.exists() && dev_int8_dec.exists() {
                        (dev_int8_enc, dev_int8_dec, dev_tok)
                    } else {
                        (dev_fp32_enc, dev_fp32_dec, dev_tok)
                    }
                }
            };

            let enc_str = encoder_path.to_str().unwrap_or("models/encoder_model.onnx");
            let dec_str = decoder_path.to_str().unwrap_or("models/decoder_model.onnx");
            let tok_str = tokenizer_path
                .to_str()
                .unwrap_or("models/tokenizer.json");

            let translator = match DefaultTranslator::load_model(enc_str, dec_str, tok_str) {
                Ok(t) => {
                    log::info!("ONNX models loaded from: {} + {}", enc_str, dec_str);
                    t
                }
                Err(e) => {
                    log::error!("Failed to load model: {}", e);
                    return Ok(());
                }
            };

            let shared_translator = Arc::new(Mutex::new(translator));

            // Manage for Tauri IPC commands
            app.manage(TranslatorState(shared_translator.clone()));

            // ── Start HTTP trigger server ──
            let app_state = Arc::new(AppState {
                translator: shared_translator,
                app: app.handle().clone(),
            });

            let router = Router::new()
                .route("/selection_translate", get(selection_translate))
                .with_state(app_state);

            let addr = SocketAddr::from(([127, 0, 0, 1], 60828));
            log::info!("HTTP trigger server on http://{}", addr);

            tauri::async_runtime::spawn(async move {
                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(l) => l,
                    Err(e) => {
                        log::error!("Failed to bind on {}: {}", addr, e);
                        return;
                    }
                };
                if let Err(e) = axum::serve(listener, router).await {
                    log::error!("HTTP server error: {}", e);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![read_selection, translate])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
