import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import BertModel
from config import VISION_MODEL_NAME, BERT_MODEL_NAME, FUSION_DIM, DROP_RATE


# Scaled dot-product attention module for fusion
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int, attn_dropout: float = 0.1):
        super().__init__()
        self.scale = dim ** -0.5

        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)

        self.norm_q = nn.LayerNorm(dim)

        # Attention weight dropout
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        is_vec = (q.dim() == 2)

        # Promote vectors to sequences for unified bmm path
        if is_vec:
            q = q.unsqueeze(1)  
        if k.dim() == 2:
            k = k.unsqueeze(1)  
        if v.dim() == 2:
            v = v.unsqueeze(1)  

        q = self.norm_q(q)

        Q = self.W_Q(q) 
        K = self.W_K(k) 
        V = self.W_V(v) 

        # Attention
        scores  = torch.bmm(Q, K.transpose(1, 2)) * self.scale 
        weights = self.attn_drop(torch.softmax(scores, dim=-1)) 
        out     = torch.bmm(weights, V)                         

        # Return to original shape
        if is_vec:
            out = out.squeeze(1)  

        return out


# Five-input fusion module from Tuan & Pham (2021)
#
# Tf    : Final text feature vector   (BERT [CLS], projected to fusion_dim)
# If    : Final image feature vector  (EfficientNetV2 GAP, projected to fusion_dim)
# RTI   : text (Tf) queries image spatial sequence (Im_seq)
# RIT   : image (If) queries BERT token sequence (Tm_seq) 
# RII   : image spatial self-attention (Im_seq)
#
# Im_seq : (B, H*W, fusion_dim) 
# Tm_seq : (B, seq_len, fusion_dim)

class _AttentionBlock(nn.Module):
    def __init__(self, fusion_dim: int, dropout: float):
        super().__init__()
        self.attn = ScaledDotProductAttention(fusion_dim, attn_dropout=0.1)

        self.fc_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(4)
        ])

        self.fc_out  = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(fusion_dim)

    def forward(self, q, k, v):
        attn_out = self.attn(q=q, k=k, v=v) 

        if attn_out.dim() == 3:
            attn_out = attn_out.mean(dim=1)   

        fc_outs = torch.stack([fc(attn_out) for fc in self.fc_layers], dim=1) 
        pooled  = fc_outs.max(dim=1).values                                    

        out = self.dropout(self.fc_out(pooled)) + attn_out  
        return self.norm(out)


class FusionModule(nn.Module):
    def __init__(self, fusion_dim: int, dropout: float = 0.1,
                 vis_seq_dim: int = None, bert_seq_dim: int = None):
        super().__init__()

        # Three attention blocks
        self.AttTI = _AttentionBlock(fusion_dim, dropout)   # text queries image spatial seq
        self.AttIT = _AttentionBlock(fusion_dim, dropout)   # image queries BERT token seq
        self.AttII = _AttentionBlock(fusion_dim, dropout)   # image spatial self-attention

        # Project spatial image features
        self.im_seq_proj = nn.Sequential(
            nn.Linear(vis_seq_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if vis_seq_dim is not None else nn.Identity()

        # Project BERT token hidden states
        self.tm_seq_proj = nn.Sequential(
            nn.Linear(bert_seq_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if bert_seq_dim is not None else nn.Identity()

        self.norm_text  = nn.LayerNorm(fusion_dim)
        self.norm_image = nn.LayerNorm(fusion_dim)

        # Projection of concatenated 5 representations
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim * 5, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Tf: torch.Tensor,       
        If: torch.Tensor,       
        Im_seq: torch.Tensor,   
        Tm_seq: torch.Tensor,    
    ) -> torch.Tensor:

        # Normalise pooled modality vectors
        Tf = self.norm_text(Tf)
        If = self.norm_image(If)

        # Project sequences to fusion_dim
        Im_seq = self.im_seq_proj(Im_seq)  
        Tm_seq = self.tm_seq_proj(Tm_seq)  

        # RTI: text vector queries real image spatial regions
        RTI = self.AttTI(q=Tf, k=Im_seq, v=Im_seq)  

        # RIT: image vector queries real BERT token sequence
        RIT = self.AttIT(q=If, k=Tm_seq, v=Tm_seq)  

        # RII: spatial image self-attention
        RII = self.AttII(q=Im_seq, k=Im_seq, v=Im_seq)  

        # Concatenate 5 representations
        fused = torch.cat([Tf, If, RTI, RIT, RII], dim=-1)   
        fused = self.fusion_proj(fused)                        
        return self.dropout(fused)


class MultimodalFakeNewsDetector(nn.Module):
    def __init__(
        self,
        vision_model_name = VISION_MODEL_NAME,
        bert_model_name   = BERT_MODEL_NAME,
        num_classes       = 2,
        fusion_dim        = FUSION_DIM,
        drop_rate         = DROP_RATE,
        mode              = "multimodal",
    ):
        super().__init__()
        self.mode = mode
        use_vision = mode in ("multimodal", "vision_only")
        use_text   = mode in ("multimodal", "text_only")

        vis_seq_dim  = None
        bert_seq_dim = None

        # Vision backbone
        if use_vision:
            self.vision_encoder = timm.create_model(
                vision_model_name, pretrained=True, num_classes=0,
                features_only=True, out_indices=(-1,),
                drop_rate=drop_rate,
            )

            vis_seq_dim = self.vision_encoder.feature_info[-1]["num_chs"]

            self.gap = nn.AdaptiveAvgPool2d(1)

            self.vis_proj = nn.Sequential(
                nn.Linear(vis_seq_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
            )
        else:
            self.vision_encoder = None
            self.vis_proj       = None
            self.gap            = None

        # Text backbone
        if use_text:
            bert_config = BertModel.from_pretrained(bert_model_name).config
            bert_config.hidden_dropout_prob          = drop_rate
            bert_config.attention_probs_dropout_prob = drop_rate
            self.bert = BertModel.from_pretrained(
                bert_model_name, config=bert_config
            )
            bert_seq_dim = bert_config.hidden_size 

            # Project [CLS] to fusion_dim  →  Tf
            self.text_proj = nn.Sequential(
                nn.Linear(bert_seq_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
            )
        else:
            self.bert      = None
            self.text_proj = None

        # Fusion
        if mode == "multimodal":
            self.fusion = FusionModule(
                fusion_dim   = fusion_dim,
                dropout      = drop_rate,
                vis_seq_dim  = vis_seq_dim,    
                bert_seq_dim = bert_seq_dim,   
            )
            fused_dim = fusion_dim * 2
        else:
            self.fusion = None
            fused_dim   = fusion_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        if self.mode == "multimodal":
            # Vision 
            feat_map = self.vision_encoder(images)[-1]
            Im_seq = feat_map.flatten(2).transpose(1, 2)
            If = self.vis_proj(self.gap(feat_map).flatten(1))

            # Text 
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            Tm_seq = bert_out.last_hidden_state 
            Tf = self.text_proj(Tm_seq[:, 0, :])

            # Fusion 
            fused = self.fusion(Tf, If, Im_seq, Tm_seq)

        elif self.mode == "vision_only":
            feat_map = self.vision_encoder(images)[-1]
            fused = self.vis_proj(self.gap(feat_map).flatten(1))

        else:  # text_only
            cls_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0, :]
            fused = self.text_proj(cls_out)

        return self.classifier(fused)

    def predict_with_confidence(self, images, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            logits = self(images, input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1)
            p_fake = probs[:, 1]; p_real = probs[:, 0]
            preds  = logits.argmax(dim=-1)
        return preds.tolist(), (p_fake * 100).tolist(), (p_real * 100).tolist()

