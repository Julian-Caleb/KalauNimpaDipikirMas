import torch
import torch.nn as nn
import timm
from transformers import BertModel
from config import VISION_MODEL, BERT_MODEL, FUSION_DIM, DROP_RATE


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        scores  = torch.bmm(q, k.transpose(1, 2)) * self.scale
        return torch.bmm(torch.softmax(scores, dim=-1), v).squeeze(1)


class FusionModule(nn.Module):
    def __init__(self, fusion_dim, dropout=0.1):
        super().__init__()
        self.AttTI   = ScaledDotProductAttention(fusion_dim)
        self.AttIT   = ScaledDotProductAttention(fusion_dim)
        self.AttII   = ScaledDotProductAttention(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Tf, If):
        Im = If; Tm = Tf
        fused = torch.cat([Tf, If,
                           self.AttTI(Tf, Im, Im),
                           self.AttIT(If, Tm, Tm),
                           self.AttII(Im, Im, Im)], dim=-1)
        return self.dropout(fused)


class MultimodalFakeNewsDetector(nn.Module):
    def __init__(self, vision_model_name=VISION_MODEL, bert_model_name=BERT_MODEL,
                 num_classes=2, fusion_dim=FUSION_DIM, drop_rate=DROP_RATE, mode="multimodal"):
        super().__init__()
        self.mode = mode
        use_vision = mode in ("multimodal", "vision_only")
        use_text   = mode in ("multimodal", "text_only")

        if use_vision:
            self.vision_encoder = timm.create_model(vision_model_name, pretrained=False,
                                                    num_classes=0, drop_rate=drop_rate)
            self.vis_proj = nn.Sequential(
                nn.Linear(self.vision_encoder.num_features, fusion_dim), nn.GELU(), nn.Dropout(drop_rate))
        else:
            self.vision_encoder = self.vis_proj = None

        if use_text:
            self.bert      = BertModel.from_pretrained(bert_model_name)
            self.text_proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, fusion_dim), nn.GELU(), nn.Dropout(drop_rate))
        else:
            self.bert = self.text_proj = None

        if mode == "multimodal":
            self.fusion = FusionModule(fusion_dim, dropout=drop_rate)
            fused_dim   = fusion_dim * 5
        else:
            self.fusion = None
            fused_dim   = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(drop_rate),
            nn.Linear(fused_dim, num_classes))

    def forward(self, images, input_ids, attention_mask):
        if self.mode == "multimodal":
            If    = self.vis_proj(self.vision_encoder(images))
            Tf    = self.text_proj(self.bert(input_ids=input_ids,
                                             attention_mask=attention_mask).last_hidden_state[:, 0, :])
            fused = self.fusion(Tf, If)
        elif self.mode == "vision_only":
            fused = self.vis_proj(self.vision_encoder(images))
        else:
            fused = self.text_proj(self.bert(input_ids=input_ids,
                                             attention_mask=attention_mask).last_hidden_state[:, 0, :])
        return self.classifier(fused)