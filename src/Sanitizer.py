import sys
import re
import string
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.setting import PROJECT_ROOT

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _factory = StopWordRemoverFactory()
    _STOPWORDS = set(_factory.get_stop_words())
except Exception:
    _STOPWORDS = set(
        [
            "yang","dan","atau","di","ke","dari","untuk","pada","dengan","itu","ini","ia","dia","kami","kita","saya","aku","anda","kamu","mereka","atau","sebagai","karena","jadi","agar","pun","lah","kah","nya","nya","kok","dong","deh","nih","tuh","loh","wkwk","br","amp","atau","ya","eh","kan","pun","dst","dll",
        ]
    )

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^0-9a-zA-Z\s]", flags=re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")

def _strip_html(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    if BeautifulSoup is not None:
        try:
            s = BeautifulSoup(x, "html.parser")
            return s.get_text(separator=" ")
        except Exception:
            return _HTML_TAG_RE.sub(" ", x)
    return _HTML_TAG_RE.sub(" ", x)

def _sanitize_text(x: str) -> list:
    if x is None:
        return []
    s = str(x)
    s = s.lower()
    s = _strip_html(s)
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _NON_ASCII_RE.sub(" ", s)
    s = _NON_ALNUM_SPACE_RE.sub(" ", s)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    tokens = [t for t in s.split(" ") if t]
    tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens

def clean_csv(input_path: Path, output_filename: str = "vibe_coding_yt_comments_clean.csv") -> Path:
    df = pd.read_csv(input_path)
    if "text" in df.columns:
        df["text"] = df["text"].apply(_sanitize_text)
    if "author" in df.columns:
        df["author"] = df["author"].astype(str).str.lower().str.replace(r"[^0-9a-zA-Z\s]", "", regex=True)
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

if __name__ == "__main__":
    in_path = PROJECT_ROOT / "data" / "vibe_coding_yt_comments.csv"
    clean_csv(in_path)