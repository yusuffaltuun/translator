use std::net::SocketAddr;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

use axum::{Router, extract::State as AxumState, routing::get};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri::WebviewWindow;

struct TranslatorState(Arc<Mutex<translator::translator::Translator>>);

struct AppState {
    translator: Arc<Mutex<translator::translator::Translator>>,
    visible: AtomicBool,
    app: AppHandle,
}

#[tauri::command]
fn read_selection() -> Result<String, String> {
    translator::read_selection()
}

#[tauri::command]
fn translate(text: String, state: State<'_, TranslatorState>) -> Result<String, String> {
    let mut translator = state.0.lock().map_err(|e| e.to_string())?;
    translator.translate(&text)
}

async fn selection_translate(
    AxumState(state): AxumState<Arc<AppState>>,
) -> axum::Json<serde_json::Value> {
    let was_visible = state.visible.fetch_xor(true, Ordering::SeqCst);

    let window: Option<WebviewWindow> = state.app.get_webview_window("main");

    if was_visible {
        if let Some(w) = &window {
            let _ = w.hide();
        }
        log::info!("overlay hidden via HTTP");
        return axum::Json(serde_json::json!({ "action": "hidden" }));
    }

    if let Some(w) = &window {
        let _ = w.show();
        let _ = w.set_focus();
    }
    log::info!("overlay shown via HTTP, reading selection…");

    match translator::read_selection() {
        Ok(text) if text.trim().is_empty() => {
            let _ = state.app.emit("translate-result", serde_json::json!({
                "source": "",
                "error": "No primary selection found. Select some text first."
            }));
            axum::Json(serde_json::json!({
                "source": "",
                "error": "No primary selection found. Select some text first."
            }))
        }
        Ok(text) => {
            let translation = state
                .translator
                .lock()
                .map_err(|e| e.to_string())
                .and_then(|mut t| t.translate(&text));

            let payload = match translation {
                Ok(tr) => serde_json::json!({
                    "source": text,
                    "translation": tr
                }),
                Err(e) => serde_json::json!({
                    "source": text,
                    "error": e
                }),
            };

            let _ = state.app.emit("translate-result", &payload);
            axum::Json(payload)
        }
        Err(e) => {
            let _ = state.app.emit("translate-result", serde_json::json!({
                "source": "",
                "error": e
            }));
            axum::Json(serde_json::json!({
                "source": "",
                "error": e
            }))
        }
    }
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

            let translator = match translator::translator::Translator::load_model(model_str, tok_str) {
                Ok(t) => {
                    log::info!("ONNX model loaded from: {}", model_str);
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
                visible: AtomicBool::new(false),
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
