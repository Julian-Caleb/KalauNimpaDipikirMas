import re
import ftfy
import emoji
from PIL import Image
from torchvision import transforms
from wordfreq import word_frequency
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

_STRIP_CHARS  = ".,!?;:\"'()[]{}-<>@#<"
_RE_NON_LATIN = re.compile(r"[^\x00-\xFF]+")
_RE_PUNCT_REP = re.compile(r"([^\w\s])\1{2,}")
_ID_THRESHOLD = 1e-6
NOISE_WORDS   = {"subscribe","follow","like","share","comment","breaking","news","live","update","official"}


def _is_id_dominant(t):
    c = t.strip(_STRIP_CHARS).lower()
    return bool(c) and word_frequency(c,"id") >= _ID_THRESHOLD and word_frequency(c,"id") > word_frequency(c,"en")

def _is_en_dominant(t):
    c = t.strip(_STRIP_CHARS).lower()
    return bool(c) and word_frequency(c,"en") >= _ID_THRESHOLD and word_frequency(c,"en") > word_frequency(c,"id")

def _is_unknown(t):
    c = t.strip(_STRIP_CHARS).lower()
    return word_frequency(c,"id") == 0 and word_frequency(c,"en") == 0

def _is_capitalized(t):
    s = t.strip(_STRIP_CHARS)
    return bool(s) and s[0].isupper()

def _is_ocr_noise(t):
    c = t.strip(_STRIP_CHARS).lower()
    if len(c) < 4: return False
    return sum(1 for ch in c if ch in "aiueo") / len(c) < 0.15 or bool(re.search(r"[^aiueo]{5,}", c))


def preprocess_ocr_text(text, min_token_len=2, min_clean_words=3):
    if not isinstance(text, str) or not text.strip(): return ""
    text   = ftfy.fix_text(text)
    text   = emoji.replace_emoji(text, replace=" ")
    text   = _RE_NON_LATIN.sub(" ", text)
    text   = _RE_PUNCT_REP.sub(r"\1", text)
    tokens = text.split()
    if not tokens: return ""

    n = len(tokens); cutoff_start = 0
    for i in range(min(max(5, n//3), n)):
        if _is_en_dominant(tokens[i]) or tokens[i].lower() in NOISE_WORDS:
            j = i
            while j < n and (_is_en_dominant(tokens[j]) or tokens[j].lower() in NOISE_WORDS
                              or (_is_unknown(tokens[j]) and _is_ocr_noise(tokens[j]))): j += 1
            cutoff_start = j; break
    tokens = tokens[cutoff_start:]
    if not tokens: return ""

    cutoff_end = len(tokens)
    for i in range(len(tokens)-1, -1, -1):
        tok = tokens[i]
        if _is_en_dominant(tok) or tok.lower() in NOISE_WORDS or (_is_unknown(tok) and _is_ocr_noise(tok)):
            cutoff_end = i
        else: break
    tokens = tokens[:cutoff_end]
    if not tokens: return ""

    tokens = [t for t in tokens if len(t.strip(_STRIP_CHARS)) >= min_token_len]
    id_pos = [i for i, t in enumerate(tokens) if _is_id_dominant(t)]
    if not id_pos: return ""

    first_id, last_id = id_pos[0], id_pos[-1]; kept = []
    for i in range(first_id, last_id+1):
        tok = tokens[i]
        if _is_id_dominant(tok):
            kept.append(tok)
        elif (_is_unknown(tok) and _is_capitalized(tok) and not _is_ocr_noise(tok)
              and tok.lower() not in NOISE_WORDS):
            if any(_is_id_dominant(tokens[j]) for j in range(first_id, i)) or \
               any(_is_id_dominant(tokens[j]) for j in range(i+1, last_id+1)):
                kept.append(tok)

    cleaned = " ".join(kept).strip()
    return cleaned if len(cleaned.split()) >= min_clean_words else ""


def pad_and_resize(image):
    w, h = image.size; sq = max(w, h)
    result = Image.new("RGB", (sq, sq), (0,0,0))
    result.paste(image, ((sq-w)//2, (sq-h)//2))
    return result.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)


eval_transform = transforms.Compose([
    transforms.Lambda(pad_and_resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])