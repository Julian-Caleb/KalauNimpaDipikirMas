import torch

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISION_MODEL   = "tf_efficientnetv2_xl.in21k"
BERT_MODEL     = "indobenchmark/indobert-base-p1"
IMG_SIZE       = 400
FUSION_DIM     = 512
NUM_ATTN_HEADS = 8
DROP_RATE      = 0.35
MAX_SEQ_LEN    = 128
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]