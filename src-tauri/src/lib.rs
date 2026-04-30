use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::Manager;
use tauri::{AppHandle, RunEvent, WindowEvent};

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: &str = "8000";
const OLLAMA_HOST: &str = "127.0.0.1:11500";
const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11500/v1";
const DEFAULT_MODEL_TAG: &str = "qwen3.5:4b";

struct BackendHandle(Mutex<Option<Child>>);
struct OllamaHandle(Mutex<Option<Child>>);

fn project_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("src-tauri must have a parent directory")
    .to_path_buf()
}

fn bundled_backend_path(app: &AppHandle) -> Option<PathBuf> {
  let candidates = [
    "_up_/dist/remedy-pdf-desktop-backend/remedy-pdf-desktop-backend",
    "remedy-pdf-desktop-backend/remedy-pdf-desktop-backend",
    "dist/remedy-pdf-desktop-backend/remedy-pdf-desktop-backend",
  ];
  for rel in candidates {
    if let Ok(path) = app.path().resolve(rel, tauri::path::BaseDirectory::Resource) {
      if path.exists() {
        return Some(path);
      }
    }
  }
  None
}

fn bundled_ollama_path(app: &AppHandle) -> Option<PathBuf> {
  let candidates: &[&str] = if cfg!(target_os = "windows") {
    &[
      "_up_/resources/ollama/windows/ollama.exe",
      "_up_/ollama/windows/ollama.exe",
      "ollama/windows/ollama.exe",
      "resources/ollama/windows/ollama.exe",
      "_up_/resources/ollama/ollama.exe",
      "ollama/ollama.exe",
    ]
  } else {
    &[
      "_up_/resources/ollama/macos/ollama",
      "_up_/ollama/macos/ollama",
      "ollama/macos/ollama",
      "resources/ollama/macos/ollama",
      "_up_/resources/ollama/ollama",
      "ollama/ollama",
    ]
  };

  for rel in candidates {
    if let Ok(path) = app.path().resolve(rel, tauri::path::BaseDirectory::Resource) {
      if path.exists() {
        return Some(path);
      }
    }
  }
  None
}

fn system_ollama_path() -> Option<PathBuf> {
  if let Ok(path) = std::env::var("PROJECT_REMEDY_OLLAMA_CMD") {
    let candidate = PathBuf::from(path);
    if candidate.exists() {
      return Some(candidate);
    }
  }

  let candidates: &[&str] = if cfg!(target_os = "windows") {
    &["C:\\Program Files\\Ollama\\ollama.exe"]
  } else {
    &[
      "/usr/local/bin/ollama",
      "/opt/homebrew/bin/ollama",
      "/Applications/Ollama.app/Contents/Resources/ollama",
    ]
  };

  for candidate in candidates {
    let path = PathBuf::from(candidate);
    if path.exists() {
      return Some(path);
    }
  }

  None
}

fn ollama_models_dir(app: &AppHandle) -> std::io::Result<PathBuf> {
  let local_data_dir = app
    .path()
    .app_local_data_dir()
    .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;
  let models_dir = local_data_dir.join("ollama").join("models");
  fs::create_dir_all(&models_dir)?;
  Ok(models_dir)
}

fn backend_data_dirs(app: &AppHandle) -> std::io::Result<(PathBuf, PathBuf)> {
  let local_data_dir = app
    .path()
    .app_local_data_dir()
    .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;
  let upload_dir = local_data_dir.join("data").join("uploads");
  let output_dir = local_data_dir.join("data").join("output");
  fs::create_dir_all(&upload_dir)?;
  fs::create_dir_all(&output_dir)?;
  Ok((upload_dir, output_dir))
}

fn configure_backend_command(command: &mut Command, app: &AppHandle) -> std::io::Result<()> {
  let models_dir = ollama_models_dir(app)?;
  let (upload_dir, output_dir) = backend_data_dirs(app)?;

  command
    .env("PROJECT_REMEDY_BACKEND_HOST", BACKEND_HOST)
    .env("PROJECT_REMEDY_BACKEND_PORT", BACKEND_PORT)
    .env("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    .env("OLLAMA_VISION_MODEL", DEFAULT_MODEL_TAG)
    .env("OLLAMA_TEXT_MODEL", DEFAULT_MODEL_TAG)
    .env("OLLAMA_MODELS", &models_dir)
    .env("UPLOAD_DIR", &upload_dir)
    .env("OUTPUT_DIR", &output_dir);

  Ok(())
}

fn configure_ollama_command(command: &mut Command, app: &AppHandle) -> std::io::Result<()> {
  let models_dir = ollama_models_dir(app)?;

  command
    .env("OLLAMA_HOST", OLLAMA_HOST)
    .env("OLLAMA_MODELS", &models_dir)
    .env("OLLAMA_KEEP_ALIVE", "10m")
    .env("OLLAMA_NUM_PARALLEL", "1")
    .env("OLLAMA_NO_CLOUD", "1")
    // Cap context to keep KV cache under ~1GB — fits on 8GB RAM systems.
    // qwen3:4b's default 262144-token context allocates ~19GB of KV alone.
    .env("OLLAMA_CONTEXT_LENGTH", "8192")
    // Quantize KV cache to halve its memory cost at negligible quality loss.
    // Requires flash-attention.
    .env("OLLAMA_FLASH_ATTENTION", "1")
    .env("OLLAMA_KV_CACHE_TYPE", "q8_0");

  Ok(())
}

fn spawn_ollama(app: &AppHandle) -> std::io::Result<Child> {
  if let Some(bundled) = bundled_ollama_path(app) {
    log::info!("Spawning bundled Ollama runtime: {}", bundled.display());
    let mut command = Command::new(bundled);
    configure_ollama_command(&mut command, app)?;
    return command.arg("serve").spawn();
  }

  if let Some(system_path) = system_ollama_path() {
    log::info!("Spawning local Ollama via {}", system_path.display());
    let mut command = Command::new(system_path);
    configure_ollama_command(&mut command, app)?;
    return command.arg("serve").spawn();
  }

  let ollama_cmd = std::env::var("PROJECT_REMEDY_OLLAMA_CMD").unwrap_or_else(|_| "ollama".into());
  log::info!("Spawning local Ollama via fallback command {ollama_cmd}");
  let mut command = Command::new(ollama_cmd);
  configure_ollama_command(&mut command, app)?;
  command.arg("serve").spawn()
}

fn spawn_backend(app: &AppHandle) -> std::io::Result<Child> {
  if let Some(bundled) = bundled_backend_path(app) {
    log::info!("Spawning bundled backend: {}", bundled.display());
    let mut command = Command::new(bundled);
    configure_backend_command(&mut command, app)?;
    return command.spawn();
  }

  let backend_cmd = std::env::var("PROJECT_REMEDY_BACKEND_CMD").unwrap_or_else(|_| "uvicorn".into());
  log::info!("Spawning dev backend via system {backend_cmd}");
  let mut command = Command::new(backend_cmd);
  configure_backend_command(&mut command, app)?;
  command
    .args([
      "backend.app.main:app",
      "--host",
      BACKEND_HOST,
      "--port",
      BACKEND_PORT,
    ])
    .current_dir(project_root())
    .spawn()
}

fn kill_stale_bundled_backends(app: &AppHandle) {
  let Some(bundled) = bundled_backend_path(app) else {
    return;
  };

  #[cfg(not(target_os = "windows"))]
  {
    let pattern = bundled.to_string_lossy().into_owned();
    let output = match Command::new("pgrep").args(["-f", &pattern]).output() {
      Ok(output) => output,
      Err(err) => {
        log::warn!("Failed to inspect stale backend processes: {err}");
        return;
      }
    };

    if !output.status.success() {
      return;
    }

    for line in String::from_utf8_lossy(&output.stdout).lines() {
      let pid = match line.trim().parse::<u32>() {
        Ok(pid) if pid != std::process::id() => pid,
        _ => continue,
      };

      log::warn!("Killing stale bundled backend process {pid}");
      let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
      thread::sleep(Duration::from_millis(300));
      let still_running = Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
      if still_running {
        let _ = Command::new("kill").args(["-KILL", &pid.to_string()]).status();
      }
    }
  }
}

fn kill_child(handle: &Mutex<Option<Child>>) {
  if let Ok(mut guard) = handle.lock() {
    if let Some(mut child) = guard.take() {
      let _ = child.kill();
      let _ = child.wait();
    }
  }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_fs::init())
    .plugin(tauri_plugin_updater::Builder::new().build())
    .manage(BackendHandle(Mutex::new(None)))
    .manage(OllamaHandle(Mutex::new(None)))
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      kill_stale_bundled_backends(app.handle());

      match spawn_ollama(app.handle()) {
        Ok(child) => {
          log::info!("Spawned Ollama runtime (pid {})", child.id());
          *app.state::<OllamaHandle>().0.lock().unwrap() = Some(child);
        }
        Err(err) => {
          log::error!("Failed to spawn Ollama runtime: {err}");
          eprintln!("Failed to spawn Ollama runtime: {err}");
        }
      }

      match spawn_backend(app.handle()) {
        Ok(child) => {
          log::info!("Spawned Python backend (pid {})", child.id());
          *app.state::<BackendHandle>().0.lock().unwrap() = Some(child);
        }
        Err(err) => {
          log::error!("Failed to spawn Python backend: {err}");
          eprintln!("Failed to spawn Python backend: {err}");
        }
      }

      Ok(())
    })
    .on_window_event(|window, event| {
      if let WindowEvent::CloseRequested { .. } = event {
        if let Some(state) = window.app_handle().try_state::<BackendHandle>() {
          kill_child(&state.0);
        }
        if let Some(state) = window.app_handle().try_state::<OllamaHandle>() {
          kill_child(&state.0);
        }
      }
    })
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|app_handle, event| {
      if let RunEvent::ExitRequested { .. } = event {
        if let Some(state) = app_handle.try_state::<BackendHandle>() {
          kill_child(&state.0);
        }
        if let Some(state) = app_handle.try_state::<OllamaHandle>() {
          kill_child(&state.0);
        }
      }
    });
}
