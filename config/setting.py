import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ENV_PATH = PROJECT_ROOT / "envs" / ".envs"
load_dotenv(dotenv_path=ENV_PATH)

def _get(name: str):
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().strip('"').strip("'")

API_KEY_RAYHAN = _get("API_KEY_RAYHAN")
API_KEY_SALAS = _get("API_KEY_SALAS")
API_KEYS = [k for k in (API_KEY_RAYHAN, API_KEY_SALAS) if k]

def get_api_key(preferred: str | None = None, index: int = 0):
    if preferred:
        v = _get(preferred)
        if v:
            return v
    if API_KEYS:
        if 0 <= index < len(API_KEYS):
            return API_KEYS[index]
        return API_KEYS[0]
    return None

# Folder Utama
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"