import re
import ftfy
import emoji
from PIL import Image
from torchvision import transforms
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

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


eval_transform = transforms.Compose([
    transforms.Lambda(pad_and_resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])