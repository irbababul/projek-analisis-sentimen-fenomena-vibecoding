import pandas as pd
import time
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.setting import API_KEYS, get_api_key, PROJECT_ROOT

# ==============================
# Config
# ==============================

output_dir = PROJECT_ROOT / "data"

VIDEO_IDS = [
    '2wwp3dKbGE8',
    '3UTMXKg47BM',
    'Mp9RTBHBv48',
    'w_eNLYWjGNY'
]

CURRENT_KEY_INDEX = 0
youtube = build("youtube", "v3", developerKey=get_api_key(index=CURRENT_KEY_INDEX))

# ==============================
# Helper: Rotate API Key
# ==============================

def rotate_key_and_rebuild():
    """
    Rotate ke API key berikutnya lalu rebuild client.
    Return True jika sukses rotate, False jika key habis.
    """
    global CURRENT_KEY_INDEX, youtube
    if CURRENT_KEY_INDEX + 1 < len(API_KEYS):
        CURRENT_KEY_INDEX += 1
        youtube = build("youtube", "v3", developerKey=get_api_key(index=CURRENT_KEY_INDEX))
        print(f"[INFO] Switch ke API key index: {CURRENT_KEY_INDEX}")
        return True
    return False

def get_vid_comments(video_id, sleep_sec=0.2):
    all_comments = []
    next_page_token = None
    while True:
        try:
            req = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText',
                order='time'
            )
            res = req.execute()
        except HttpError as e:
            if rotate_key_and_rebuild():
                continue
            print(f"[ERROR] video {video_id}: {e}")
            break
        for item in res.get('items', []):
            top_comment = item['snippet']['topLevelComment']
            top_snip = top_comment['snippet']
            top_id = top_comment['id']
            all_comments.append({
                'video_id': video_id,
                'comment_id': top_id,
                'parent_id': None,
                'author': top_snip.get('authorDisplayName'),
                'text': top_snip.get('textDisplay'),
                'like_count': top_snip.get('likeCount'),
                'published_at': top_snip.get('publishedAt'),
                'updated_at': top_snip.get('updatedAt'),
                'reply_count': item['snippet'].get('totalReplyCount', 0),
            })
            replies_inc = item.get('replies', {}).get('comments', [])
            for reply in replies_inc:
                r_snip = reply['snippet']
                all_comments.append({
                    'video_id': video_id,
                    'comment_id': reply['id'],
                    'parent_id': top_id,
                    'author': r_snip.get('authorDisplayName'),
                    'text': r_snip.get('textDisplay'),
                    'like_count': r_snip.get('likeCount'),
                    'published_at': r_snip.get('publishedAt'),
                    'updated_at': r_snip.get('updatedAt'),
                    'reply_count': None,
                })
            total_reply = item['snippet'].get('totalReplyCount', 0)
            if total_reply > len(replies_inc):
                all_comments.extend(get_all_rep(top_id))
        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break
        time.sleep(sleep_sec)
    return all_comments

# ==============================
# GET ALL REPLIES
# ==============================

def get_all_rep(comment_id, sleep_sec=1):
    """
    Ambil semua replies dari 1 top-level comment dengan comments.list.
    """

    replies = []
    next_page_token = None

    while True:
        try:
            req = youtube.comments().list(
                part='snippet',
                parentId=comment_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText',
            )
            res = req.execute()

        except HttpError as e:
            print(f"[ERROR] replies parent {comment_id}: {e}")
            break

        for item in res.get("items", []):
            snip = item["snippet"]
            replies.append({
                "video_id": snip.get("videoId"),
                "comment_id": item["id"],
                "parent_id": comment_id,
                "author": snip.get("authorDisplayName"),
                "text": snip.get("textDisplay"),
                "like_count": snip.get("likeCount"),
                "published_at": snip.get("publishedAt"),
                "updated_at": snip.get("updatedAt"),
                "reply_count": None
            })

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(sleep_sec)

    return replies


def get_comments_for_videos(video_ids):
    """
    Function untuk mengambil (looping) komentar dari list video_ids
    """
    all_data = []
    for vid in video_ids:
        print(f"== Mengambil komentar video: {vid}")
        video_comments = get_vid_comments(vid)
        print(f"  ==> dapat {len(video_comments)} komentar (termasuk replies)")
        all_data.extend(video_comments)
    return all_data

def save_to_csv(data, filename="vibe_coding_yt_comments.csv"):
    """
    Function untuk menyimpan data ke file CSV
    """
    if not data:
        print("Tidak ada data untuk disimpan")
        return

    df = pd.DataFrame(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"==> data disimpan ke {path}")

if __name__ == "__main__":
    data = get_comments_for_videos(VIDEO_IDS)
    save_to_csv(data)
