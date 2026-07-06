import io
import re
import ftfy
import emoji
from PIL import Image, ImageChops
from torchvision import transforms
from config import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ELA_QUALITY,
    ELA_AMPLIFY,
    ELA_BLEND_ALPHA,
)

_RE_PUNCT_REPEAT = re.compile(r"([^\w\s])\1{2,}")

def preprocess_ocr_text(text: str) -> str:
    
    # Validate input
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Fix encoding problems
    text = ftfy.fix_text(text)
    
    # Remove emoji
    text = emoji.replace_emoji(text, replace=" ")
    
    # Collapse repeated punctuation
    text = _RE_PUNCT_REPEAT.sub(r"\1", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Pad image to square with black border, then resize.
def pad_and_resize(image: Image.Image, target_size: int = IMG_SIZE) -> Image.Image:
    w, h = image.size
    sq   = max(w, h)

    result = Image.new("RGB", (sq, sq), (0, 0, 0))
    result.paste(image, ((sq - w) // 2, (sq - h) // 2))

    return result.resize((target_size, target_size), Image.Resampling.BILINEAR)


# Error Level Analysis (ELA)
def compute_ela(
    image: Image.Image,
    quality: int = ELA_QUALITY,
    amplify: int = ELA_AMPLIFY,
) -> Image.Image:
    image = image.convert("RGB")

    # Recompress image
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    recompressed = Image.open(buf).convert("RGB")

    # Compute difference
    ela_map = ImageChops.difference(image, recompressed)

    # Amplify residual
    bands = ela_map.split()
    amplified = []

    for band in bands:
        lo, hi = band.getextrema()

        max_val = hi if hi != 0 else 1
        scale   = min(255.0 / max_val * amplify, 255.0)

        amplified.append(
            band.point(lambda p, s=scale: int(p * s))
        )

    return Image.merge("RGB", amplified)

def ela_blend(image: Image.Image, alpha: float = ELA_BLEND_ALPHA) -> Image.Image:
    ela_map = compute_ela(image)

    return Image.blend(
        image.convert("RGB"),
        ela_map,
        alpha=1 - alpha,
    )


eval_transform = transforms.Compose([
    transforms.Lambda(pad_and_resize),
    transforms.Lambda(ela_blend),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])