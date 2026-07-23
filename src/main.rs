use std::env;

#[derive(Debug, PartialEq)]
enum DisplayServer {
    Wayland,
    X11,
    Unknown,
}

fn detect_display_server() -> DisplayServer {
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

fn main() {
    let server = detect_display_server();
    println!("Display server detected: {:?}", server);
}
