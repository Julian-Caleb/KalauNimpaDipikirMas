import torch
import torch.nn as nn
import timm
from transformers import BertModel
from config import VISION_MODEL, BERT_MODEL, FUSION_DIM, DROP_RATE, NUM_ATTN_HEADS


# Scaled dot-product attention with learnable W_Q, W_K, W_V projections.
# Follows Tuan & Pham (2021): Q = q × W_Q, K = k × W_K, V = v × W_V
# Attention(Q, K, V) = softmax((Q × K^T) / sqrt(d_k)) × V
#
# Supports both:
#   - vector input  (B, D)        → unsqueeze to (B, 1, D) for single-token attention
#   - sequence input (B, seq, D)  → used directly for region-level attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5

        # Learnable projection matrices W_Q, W_K, W_V (Tuan & Pham 2021, Sec. III-C)
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)

        # Pre-LN on Q before projection — stabilises training (not in paper, added for stability)
        self.norm_q = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        is_vec = (q.dim() == 2)

        # Promote vectors to sequences for unified bmm path
        if is_vec:
            q = q.unsqueeze(1)   # (B, 1, D)
        if k.dim() == 2:
            k = k.unsqueeze(1)   # (B, 1, D)
        if v.dim() == 2:
            v = v.unsqueeze(1)   # (B, 1, D)

        # Pre-LN on Q
        q = self.norm_q(q)

        # Learnable projections: Q = q × W_Q, K = k × W_K, V = v × W_V
        Q = self.W_Q(q)   # (B, seq_q, D)
        K = self.W_K(k)   # (B, seq_k, D)
        V = self.W_V(v)   # (B, seq_v, D)

        # Attention: softmax((Q × K^T) / sqrt(d_k)) × V
        scores  = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # (B, seq_q, seq_k)
        weights = torch.softmax(scores, dim=-1)                   # (B, seq_q, seq_k)
        out     = torch.bmm(weights, V)                           # (B, seq_q, D)

        # Return to original shape
        if is_vec:
            out = out.squeeze(1)   # (B, D)

        return out

# Five-input fusion module from Tuan & Pham (2021), Section III-C and III-D.
#
# Tf    : Final text feature vector  (BERT [CLS], projected to fusion_dim)
# If    : Final image feature vector (EfficientNetV2, projected to fusion_dim)
# RTI   : text (Tf) queries image (Im) — cross-modal
# RIT   : image (If) queries text (Tm) — cross-modal
# RII   : image self-attention
#
# Per paper: after each attention block, output is passed through FC layers,
# then a residual connection is added, then LayerNorm is applied (Post-LN).

class _AttentionBlock(nn.Module):
    def __init__(self, fusion_dim: int, dropout: float):
        super().__init__()
        self.attn = ScaledDotProductAttention(fusion_dim)

        # 4 parallel FC layers (paper: "four different fully connected layers with size 32")
        self.fc_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Dropout(dropout))
            for _ in range(4)
        ])

        # 1 additional FC after max-pool + residual (paper: "pass it into another fully connected layer")
        self.fc_out  = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # LayerNorm on the output of the attention block 
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, q, k, v):
        attn_out = self.attn(q=q, k=k, v=v)   # (B, D)  — residual base

        # 4 FC projections stacked along a new dim for max-pool
        fc_outs = torch.stack([fc(attn_out) for fc in self.fc_layers], dim=1)  # (B, 4, D)
        pooled  = fc_outs.max(dim=1).values                                     # (B, D)

        # Additional FC + residual connection from attention output
        out = self.dropout(self.fc_out(pooled)) + attn_out   # (B, D)

        # Post-LN
        return self.norm(out)


class FusionModule(nn.Module):
    def __init__(self, fusion_dim: int, dropout: float = 0.1):
        super().__init__()
        # Three attention blocks, each with full post-attention processing
        self.AttTI = _AttentionBlock(fusion_dim, dropout)   # text queries image
        self.AttIT = _AttentionBlock(fusion_dim, dropout)   # image queries text
        self.AttII = _AttentionBlock(fusion_dim, dropout)   # image self-attention

        # LayerNorm on individual modalities before fusion prevents one modality
        # from dominating the attention scores (implementation choice, not in paper).
        self.norm_text  = nn.LayerNorm(fusion_dim)
        self.norm_image = nn.LayerNorm(fusion_dim)

        # Projection of concatenated 5 representations
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim * 5, fusion_dim * 2),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Tf: torch.Tensor,   # final text feature  (B, D)
        If: torch.Tensor,   # final image feature (B, D)
    ) -> torch.Tensor:

        # Normalise individual modalities (implementation choice for stability)
        Tf = self.norm_text(Tf)
        If = self.norm_image(If)

        # Im and Tm are approximated by If and Tf respectively.
        # In the original paper Im is a 2D spatial tensor from VGG-19 (49 × 32).
        # Here EfficientNetV2 outputs a pooled vector, so Im = If is a known simplification.
        Im = If
        Tm = Tf

        RTI = self.AttTI(q=Tf, k=Im, v=Im)   # (B, D) — text queries image regions
        RIT = self.AttIT(q=If, k=Tm, v=Tm)   # (B, D) — image queries text
        RII = self.AttII(q=Im, k=Im, v=Im)   # (B, D) — image self-attention

        # Concatenate 5 representations (Tuan & Pham 2021, Sec. III-D)
        fused = torch.cat([Tf, If, RTI, RIT, RII], dim=-1)   # (B, D*5)
        fused = self.fusion_proj(fused)                       # (B, D*2)
        return self.dropout(fused)


class MultimodalFakeNewsDetector(nn.Module):
    def __init__(
        self,
        vision_model_name = VISION_MODEL,
        bert_model_name   = BERT_MODEL,
        num_classes       = 2,
        fusion_dim        = FUSION_DIM,
        num_attn_heads    = NUM_ATTN_HEADS,
        drop_rate         = DROP_RATE,
        mode              = "multimodal",
    ):
        super().__init__()
        self.mode = mode
        use_vision = mode in ("multimodal", "vision_only")
        use_text   = mode in ("multimodal", "text_only")

        # Vision backbone
        if use_vision:
            self.vision_encoder = timm.create_model(
                vision_model_name, pretrained=True, num_classes=0,
                drop_rate=drop_rate,
            )
            vis_dim = self.vision_encoder.num_features
            self.vis_proj = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(vis_dim, fusion_dim),
            )
        else:
            self.vision_encoder = None
            self.vis_proj       = None

        # Text backbone
        if use_text:
            bert_config = BertModel.from_pretrained(bert_model_name).config
            bert_config.hidden_dropout_prob          = drop_rate
            bert_config.attention_probs_dropout_prob = drop_rate
            self.bert = BertModel.from_pretrained(
                bert_model_name, config=bert_config
            )
            bert_dim = bert_config.hidden_size

            # Freeze BERT pooler
            # for p in self.bert.pooler.parameters():
            #     p.requires_grad = False

            self.text_proj = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(bert_dim, fusion_dim),
            )
        else:
            self.bert      = None
            self.text_proj = None

        # Fusion
        if mode == "multimodal":
            self.fusion = FusionModule(fusion_dim=fusion_dim, dropout=drop_rate)
            fused_dim   = fusion_dim * 2
        else:
            self.fusion = None
            fused_dim   = fusion_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        if self.mode == "multimodal":
            If    = self.vis_proj(self.vision_encoder(images))
            Tf    = self.text_proj(
                self.bert(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state[:, 0, :]
            )
            fused = self.fusion(Tf, If)

        elif self.mode == "vision_only":
            fused = self.vis_proj(self.vision_encoder(images))

        else:
            cls_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
            fused = self.text_proj(cls_out)

        return self.classifier(fused)

    def predict_with_confidence(self, images, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            logits = self(images, input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1)
            p_fake = probs[:, 0]; p_real = probs[:, 1]
            preds  = logits.argmax(dim=-1)
        return preds.tolist(), (p_fake * 100).tolist(), (p_real * 100).tolist()

