"""
GroqKeyManager.py
-----------------
Utility untuk manage dan rotate Groq API keys dengan error handling.

Digunakan oleh:
- AutoLabelSentiment.py (untuk auto-labeling)
- testGroqKeys.py (untuk testing)
"""

import os
import ast
from pathlib import Path
from dotenv import load_dotenv


def load_groq_env(env_path: Path | str | None = None) -> None:
    """Load environment variables dari .envs file."""
    if env_path is None:
        env_path = Path(__file__).resolve().parents[1] / "envs" / ".envs"
    load_dotenv(dotenv_path=env_path)


def collect_groq_keys() -> list[str]:
    """
    Collect Groq API keys dari environment variables.
    
    Mendukung format:
    - Semicolon-separated: API_KEY_GROQ="key1;key2;key3"
    - Python list: API_KEY_GROQ="[key1, key2, key3]"
    - Single key: API_KEY_GROQ="key1"
    
    Returns:
        list[str]: List of unique API keys, atau empty list jika tidak ada
    """
    keys: list[str] = []
    
    for k, v in os.environ.items():
        if k.startswith("API_KEY_GROQ") or k.startswith("GROQ_API_KEY"):
            if v:
                v_clean = v.strip().strip('"').strip("'")
                
                if v_clean.startswith("[") and v_clean.endswith("]"):
                    # Parse sebagai Python list syntax
                    try:
                        parsed = ast.literal_eval(v_clean)
                        if isinstance(parsed, list):
                            keys.extend([str(item).strip() for item in parsed if item])
                    except (ValueError, SyntaxError):
                        pass
                elif ";" in v_clean:
                    # Parse sebagai semicolon-separated values
                    keys.extend([item.strip() for item in v_clean.split(";") if item.strip()])
                else:
                    # Single key
                    if v_clean:
                        keys.append(v_clean)
    
    # Deduplicate
    seen = set()
    uniq = []
    for k in keys:
        if k and k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq
