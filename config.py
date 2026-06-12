import torch

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISION_MODEL_NAME   = "tf_efficientnetv2_l.in21k"
BERT_MODEL_NAME     = "indobenchmark/indobert-base-p1"
IMG_SIZE            = 400
FUSION_DIM          = 512
DROP_RATE           = 0.50
MAX_SEQ_LEN         = 256
IMAGENET_MEAN       = [0.485, 0.456, 0.406]
IMAGENET_STD        = [0.229, 0.224, 0.225]