import torch
import torch.nn as nn
import torch.nn.functional as F
from wenet.transformer.encoder_cat import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling, BatchNorm1d
from wenet.utils.mask import make_pad_mask


class EnhancedConv1d(nn.Module):
    """Enhanced 1D convolution block."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, groups=1):
        super(EnhancedConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, groups=groups
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU(inplace=True) 
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MultiScaleSemanticFusion(nn.Module):
    """Multi-scale semantic fusion module."""

    def __init__(self, channels, num_scales=3, reduction=16):
        super(MultiScaleSemanticFusion, self).__init__()
        self.channels = channels
        self.num_scales = num_scales

        # Multi-scale semantic extraction convs
        self.semantic_convs = nn.ModuleList([
            nn.Sequential(
                EnhancedConv1d(channels, channels, kernel_size=3 + i * 4, padding=1 + i * 2),
                EnhancedConv1d(channels, channels, kernel_size=1)  # Extra 1x1 conv for stronger expressiveness
            ) for i in range(num_scales)
        ])

        # Semantic attention
        self.semantic_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels * num_scales, channels * num_scales // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(channels * num_scales // reduction, num_scales),
            nn.Softmax(dim=1)
        )

        # Normalized weight parameters
        self.additive_weights = nn.Parameter(torch.ones(3))  # α, β, γ - additive fusion weights
        self.multiplicative_weights = nn.Parameter(torch.ones(2))  # δ, ε - multiplicative fusion weights
        self.multiplicative_scale = nn.Parameter(torch.tensor(2.0))
        self.fusion_norm = nn.BatchNorm1d(channels)

    def forward(self, low_feat, mid_feat, high_feat):
        """
        Args:
            low_feat: [B, C, T] Low-level semantic feature (detail-rich)
            mid_feat: [B, C, T] Mid-level semantic feature (balanced)
            high_feat: [B, C, T] High-level semantic feature (strong semantics)
        Returns:
            fused_feat: [B, C, T] Fused feature
        """
        batch_size, channels, time_steps = low_feat.shape

        # 1) Multi-scale semantic feature extraction
        semantic_features = []

        # Multi-scale convolutions
        for feat in [low_feat, mid_feat, high_feat]:
            scale_features = []
            for conv in self.semantic_convs:
                scale_feat = conv(feat)  # [B, C, T]
                scale_features.append(scale_feat)

            # Concatenate multi-scale features: [B, C*num_scales, T]
            multi_scale_feat = torch.cat(scale_features, dim=1)
            semantic_features.append(multi_scale_feat)

        # 2) Semantic attention weighted fusion
        # Compute attention weights for each semantic level
        attention_weights = []
        for feat in semantic_features:
            # [B, C*num_scales, T] -> [B, num_scales]
            attn = self.semantic_attention(feat)
            attention_weights.append(attn)

        # 3) Weighted fusion of multi-scale features
        weighted_features = []
        for feat, weights in zip(semantic_features, attention_weights):
            # Split into scales: [B, C*num_scales, T] -> num_scales * [B, C, T]
            scale_feats = torch.chunk(feat, self.num_scales, dim=1)

            # Apply attention weights
            weighted_feat = sum(w.unsqueeze(-1).unsqueeze(-1) * scale_f
                                for w, scale_f in zip(weights.unbind(1), scale_feats))
            weighted_features.append(weighted_feat)

        low_semantic, mid_semantic, high_semantic = weighted_features

        # 4) Separately-normalized semantic-level fusion
        # Normalize additive fusion weights
        additive_weights = F.softmax(self.additive_weights, dim=0)
        α, β, γ = additive_weights

        # Normalize multiplicative fusion weights
        multiplicative_weights = F.softmax(self.multiplicative_weights, dim=0)
        δ, ε = multiplicative_weights

        # Additive fusion: linear combination of semantic levels
        additive_fusion = (α * low_semantic +
                           β * mid_semantic +
                           γ * high_semantic)

        k = torch.sigmoid(self.multiplicative_scale)
        multiplicative_fusion = k * (δ * low_semantic * mid_semantic + ε * low_semantic * high_semantic)

        fused_feat = additive_fusion + multiplicative_fusion
        fused_feat = self.fusion_norm(fused_feat)

        return fused_feat

    def get_fusion_equation(self):
        """Return the fusion equation string."""
        additive_weights = F.softmax(self.additive_weights, dim=0)
        multiplicative_weights = F.softmax(self.multiplicative_weights, dim=0)

        α, β, γ = additive_weights.detach().cpu().numpy()
        δ, ε = multiplicative_weights.detach().cpu().numpy()

        k = float(torch.sigmoid(self.multiplicative_scale).detach().cpu().item())
        equation = f"F_out = {α:.3f}·Low + {β:.3f}·Mid + {γ:.3f}·High + "
        equation += f"{k:.3f}·({δ:.3f}·(Low⊙Mid) + {ε:.3f}·(Low⊙High))"

        return equation


class SemanticAwareAttention(nn.Module):
    """Semantic-aware attention module."""

    def __init__(self, channels, num_heads=8):
        super(SemanticAwareAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        # Semantic gating
        self.semantic_gate = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(channels, channels)

    def forward(self, low_feat, mid_feat, high_feat):
        """Attention fusion based on semantic similarity."""
        batch_size, channels, time_steps = low_feat.shape

        # Convert shape: [B, C, T] -> [B, T, C]
        low_feat = low_feat.permute(0, 2, 1)
        mid_feat = mid_feat.permute(0, 2, 1)
        high_feat = high_feat.permute(0, 2, 1)

        # Use high-level features as query, low-level features as key/value (semantic guidance)
        Q = self.query(high_feat)  # [B, T, C]
        K = self.key(low_feat)  # [B, T, C]
        V = self.value(low_feat)  # [B, T, C]

        # Multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, channels)

        # Semantic gating fusion
        semantic_context = torch.cat([low_feat, mid_feat, high_feat], dim=-1)
        gate_weights = self.semantic_gate(semantic_context)

        # Gated output
        output = gate_weights * attn_output + (1 - gate_weights) * low_feat
        output = self.out_proj(output)

        # Convert back to [B, C, T]
        output = output.permute(0, 2, 1)

        return output


class EnhancedSemanticFeaturePyramid(nn.Module):
    """Enhanced Semantic Feature Pyramid Network (ES-FPN)."""

    def __init__(self, in_channels, out_channels=256, use_attention=True):
        super(EnhancedSemanticFeaturePyramid, self).__init__()

        self.out_channels = out_channels
        self.use_attention = use_attention

        # Semantic projection layers (using enhanced convs)
        self.semantic_projection = nn.ModuleList([
            nn.Sequential(
                EnhancedConv1d(in_channels, out_channels, 3, padding=1),
                EnhancedConv1d(out_channels, out_channels, 1)  # Extra 1x1 conv for stronger expressiveness
            ) for _ in range(3)  # Low / mid / high semantic levels
        ])

        # Multi-scale semantic fusion
        self.semantic_fusion = MultiScaleSemanticFusion(out_channels)

        # Semantic-aware attention
        if use_attention:
            self.semantic_attention = SemanticAwareAttention(out_channels)

        # Semantic refinement convs (using enhanced convs)
        self.refinement_conv = nn.Sequential(
            EnhancedConv1d(out_channels, out_channels, 3, padding=1),
            EnhancedConv1d(out_channels, out_channels, 3, padding=1),
            EnhancedConv1d(out_channels, out_channels, 1)  # Final 1x1 conv
        )

        # Residual path (using enhanced convs)
        self.residual_conv = nn.Sequential(
            EnhancedConv1d(in_channels, out_channels, 1),
            EnhancedConv1d(out_channels, out_channels, 1)
        )

        # Separately-normalized output weights
        self.semantic_weights = nn.Parameter(torch.ones(1))  # semantic fusion weight
        self.attention_weights = nn.Parameter(torch.ones(1))  # attention fusion weight

    def forward(self, feat_low, feat_mid, feat_high):
        # 1) Semantic projection
        proj_low = self.semantic_projection[0](feat_low)
        proj_mid = self.semantic_projection[1](feat_mid)
        proj_high = self.semantic_projection[2](feat_high)

        # 2) Multi-scale semantic fusion
        semantic_fused = self.semantic_fusion(proj_low, proj_mid, proj_high)

        # 3) Semantic-aware attention
        if self.use_attention:
            attention_fused = self.semantic_attention(proj_low, proj_mid, proj_high)

            # Separately-normalized adaptive fusion
            fusion_weights = F.softmax(torch.stack([self.semantic_weights, self.attention_weights]), dim=0)
            w_semantic, w_attention = fusion_weights

            fused_feat = w_semantic * semantic_fused + w_attention * attention_fused
        else:
            fused_feat = semantic_fused

        # 4) Semantic feature refinement
        refined_feat = self.refinement_conv(fused_feat)

        # 5) Residual connection
        if feat_low.size(1) == self.out_channels:
            residual = feat_low
        else:
            residual = self.residual_conv(feat_low)

        output = refined_feat + residual

        return output

    def get_fusion_info(self):
        """Return fusion information."""
        semantic_eq = self.semantic_fusion.get_fusion_equation()
        return f"Semantic fusion: {semantic_eq}"


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module."""

    def __init__(self, channels, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(channels, channels)
        self.value_proj = nn.Linear(channels, channels)

        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),  # SiLU activation
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, feat1, feat2):
        batch_size, channels, time_steps = feat1.size()

        feat1 = feat1.permute(0, 2, 1)
        feat2 = feat2.permute(0, 2, 1)

        residual = feat1
        feat1 = self.norm1(feat1)
        feat2 = self.norm1(feat2)

        Q = self.query_proj(feat1)
        K = self.key_proj(feat2)
        V = self.value_proj(feat2)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, channels)

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        attn_output = residual + attn_output
        output = self.norm2(attn_output)
        output = output + self.ffn(output)

        output = output.permute(0, 2, 1)
        return output


class SemanticAwareConformer(nn.Module):
    """Semantic-aware Conformer model."""

    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192,
                 input_layer="conv2d2", pos_enc_layer_type="rel_pos",
                 apply_layer_balance_to_original: bool = False):
        super(SemanticAwareConformer, self).__init__()
        self.num_blocks = num_blocks
        # False: only the semantic branch uses per-layer LayerNorm+scale; the original concat branch uses raw layer outputs
        # True: apply layer balancing to both branches
        self.apply_layer_balance_to_original = apply_layer_balance_to_original

        # Conformer encoder
        self.conformer = ConformerEncoder(
            input_size=n_mels,
            num_blocks=num_blocks,
            output_size=output_size,
            input_layer=input_layer,
            pos_enc_layer_type=pos_enc_layer_type
        )

        # Pooling and projection
        self.pooling = AttentiveStatisticsPooling(output_size * num_blocks)
        self.bn = BatchNorm1d(input_size=output_size * num_blocks * 2)

        # Cross-attention fusion
        self.cross_attn_fuser1 = CrossAttentionFusion(output_size)
        self.cross_attn_fuser2 = CrossAttentionFusion(output_size)
        self.cross_attn_fuser3 = CrossAttentionFusion(output_size)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(output_size) for _ in range(num_blocks)])
        self.layer_scales = nn.Parameter(torch.ones(num_blocks))

        # Shape: [3, num_blocks]. After softmax over dim=1, each row represents layer weights for low/mid/high
        self.semantic_group_logits = nn.Parameter(torch.zeros(3, num_blocks))
        self._init_group_logits()

        # Enhanced semantic feature pyramid
        self.semantic_fpn = EnhancedSemanticFeaturePyramid(output_size, output_size)

        # Semantic pooling
        self.semantic_pooling = AttentiveStatisticsPooling(output_size)
        self.semantic_bn = BatchNorm1d(input_size=output_size * 2)

        # Final fusion: directly consumes all pooled statistics (no bottleneck)
        self.final_fc = nn.Linear(output_size * num_blocks * 2 + output_size * 2, embedding_dim)
        self.emb_bn = nn.BatchNorm1d(embedding_dim, affine=False)

        self.output_size = output_size

    def _init_group_logits(self):
        """Initialize grouping logits close to an intuitive low/mid/high split; still learnable during training."""
        with torch.no_grad():
            logits = torch.full((3, self.num_blocks), -2.0)
            if self.num_blocks >= 3:
                split1 = self.num_blocks // 3
                split2 = (2 * self.num_blocks) // 3
                logits[0, :split1] = 2.0
                logits[1, split1:split2] = 2.0
                logits[2, split2:] = 2.0
            else:
                logits[:, :] = 1.0
            self.semantic_group_logits.copy_(logits)

    def _apply_layer_balance(self, x, layer_idx):
        """Apply per-layer balancing: LayerNorm + learnable scale. Input/Output: [B, C, T]."""
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = self.layer_norms[layer_idx](x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        return x * self.layer_scales[layer_idx]

    def _build_semantic_groups(self, layer_outputs):
        """Aggregate all layers into low/mid/high semantic features using learnable weights.

        layer_outputs: List[[B, C, T]], len=num_blocks
        """
        # [3, num_blocks]
        group_weights = F.softmax(self.semantic_group_logits, dim=1)

        # [num_blocks, B, C, T]
        stacked = torch.stack(layer_outputs, dim=0)
        grouped = []
        for g in range(3):
            w = group_weights[g].view(self.num_blocks, 1, 1, 1)
            grouped_feat = torch.sum(w * stacked, dim=0)  # [B, C, T]
            grouped.append(grouped_feat)
        return grouped, group_weights

    def forward(self, feat):
        # Preprocessing
        feat = feat.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens * feat.shape[1]).int()

        masks = ~make_pad_mask(lens).unsqueeze(1)
        if self.conformer.global_cmvn is not None:
            feat = self.conformer.global_cmvn(feat)
        xs, pos_emb, masks = self.conformer.embed(feat, masks)
        mask_pad = masks

        # Process each layer: split into an "original" path and a "semantic" path
        conformer_outputs_raw = []
        conformer_outputs_sem = []
        x = xs
        for i in range(self.num_blocks):
            x, masks, _ = self.conformer.encoders[i](x, masks, pos_emb, mask_pad)
            layer_feat_raw = x.permute(0, 2, 1)  # [B, C, T]
            layer_feat_balanced = self._apply_layer_balance(layer_feat_raw, i)
            conformer_outputs_raw.append(layer_feat_raw)
            conformer_outputs_sem.append(layer_feat_balanced)

        if self.apply_layer_balance_to_original:
            original_stack = conformer_outputs_sem
        else:
            original_stack = conformer_outputs_raw

        # Original approach: concatenate features from all blocks
        original_feat = torch.cat(original_stack, dim=1)
        original_pooled = self.pooling(original_feat)
        original_pooled = self.bn(original_pooled)
        original_pooled = original_pooled.permute(0, 2, 1)  # [B, 1, C1]

        # Semantic-level fusion
        (group_low, group_mid, group_high), _ = self._build_semantic_groups(conformer_outputs_sem)

        # Cross-group semantic interaction via cross-attention
        fused_low = self.cross_attn_fuser1(group_low, group_mid)
        fused_mid = self.cross_attn_fuser2(group_mid, group_high)
        fused_high = self.cross_attn_fuser3(group_high, group_low)

        # Cross-level fusion via the enhanced semantic feature pyramid
        semantic_feat = self.semantic_fpn(fused_low, fused_mid, fused_high)

        # Semantic pooling
        semantic_pooled = self.semantic_pooling(semantic_feat)
        semantic_pooled = self.semantic_bn(semantic_pooled)
        semantic_pooled = semantic_pooled.permute(0, 2, 1)  # [B, 1, C2]

        # Directly concatenate high-dimensional statistics
        combined = torch.cat([original_pooled, semantic_pooled], dim=2)
        final_embedding = self.final_fc(combined).squeeze(1)
        final_embedding = self.emb_bn(final_embedding)
        final_embedding = F.normalize(final_embedding, dim=1)

        return final_embedding

    def forward_with_intermediates(self, feat):
        # Preprocessing
        feat = feat.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens * feat.shape[1]).int()

        masks = ~make_pad_mask(lens).unsqueeze(1)
        if self.conformer.global_cmvn is not None:
            feat = self.conformer.global_cmvn(feat)
        xs, pos_emb, masks = self.conformer.embed(feat, masks)
        mask_pad = masks

        conformer_outputs_raw = []
        conformer_outputs_sem = []
        x = xs
        for i in range(self.num_blocks):
            x, masks, _ = self.conformer.encoders[i](x, masks, pos_emb, mask_pad)
            layer_feat_raw = x.permute(0, 2, 1)  # [B, C, T]
            layer_feat_balanced = self._apply_layer_balance(layer_feat_raw, i)
            conformer_outputs_raw.append(layer_feat_raw)
            conformer_outputs_sem.append(layer_feat_balanced)

        if self.apply_layer_balance_to_original:
            original_stack = conformer_outputs_sem
        else:
            original_stack = conformer_outputs_raw

        # Original branch
        original_feat = torch.cat(original_stack, dim=1)
        original_pooled = self.pooling(original_feat)
        original_pooled = self.bn(original_pooled)
        original_pooled = original_pooled.permute(0, 2, 1)

        # Semantic branch
        (group_low, group_mid, group_high), group_weights = self._build_semantic_groups(conformer_outputs_sem)
        fused_low = self.cross_attn_fuser1(group_low, group_mid)
        fused_mid = self.cross_attn_fuser2(group_mid, group_high)
        fused_high = self.cross_attn_fuser3(group_high, group_low)
        semantic_feat = self.semantic_fpn(fused_low, fused_mid, fused_high)

        semantic_pooled = self.semantic_pooling(semantic_feat)
        semantic_pooled = self.semantic_bn(semantic_pooled)
        semantic_pooled = semantic_pooled.permute(0, 2, 1)

        # Final fusion
        combined = torch.cat([original_pooled, semantic_pooled], dim=2)
        final_embedding = self.final_fc(combined).squeeze(1)
        final_embedding = self.emb_bn(final_embedding)
        final_embedding = F.normalize(final_embedding, dim=1)

        return {
            # For layer-norm / L0·L2·L4 PCA visualization, use raw Conformer layers to avoid confusion with the "original" branch semantics
            "conformer_outputs": conformer_outputs_raw,
            "group_low": group_low,
            "group_mid": group_mid,
            "group_high": group_high,
            "fused_low": fused_low,
            "fused_mid": fused_mid,
            "fused_high": fused_high,
            "semantic_feat": semantic_feat,
            "original_pooled": original_pooled.squeeze(1),
            "semantic_pooled": semantic_pooled.squeeze(1),
            "final_embedding": final_embedding,
            "group_weights": group_weights,
        }

    def get_fusion_info(self):
        """Get fusion information."""
        # Get semantic fusion information
        semantic_info = self.semantic_fpn.get_fusion_info()

        final_fusion_info = "\nFinal fusion: directly concatenate high-dimensional statistics (no bottleneck)"

        # Print learnable grouping info (top-2 layers per group)
        group_w = F.softmax(self.semantic_group_logits, dim=1).detach().cpu()
        group_names = ["Low", "Mid", "High"]
        group_info = []
        for gi, name in enumerate(group_names):
            topv, topi = torch.topk(group_w[gi], k=min(2, self.num_blocks))
            parts = [f"L{int(i)}:{float(v):.3f}" for v, i in zip(topv, topi)]
            group_info.append(f"{name}[{' '.join(parts)}]")
        grouping_desc = "\nLearnable grouping top layers: " + " | ".join(group_info)

        bal_mode = "Both branches balanced" if self.apply_layer_balance_to_original else "Semantic branch only"
        balance_desc = f"\nInter-layer balance: {bal_mode}"

        return semantic_info + final_fusion_info + grouping_desc + balance_desc


def conformer_cat(n_mels=80, num_blocks=6, output_size=256, embedding_dim=192,
                  input_layer="conv2d2", pos_enc_layer_type="rel_pos",
                  apply_layer_balance_to_original: bool = False):
    """Create a semantic-aware Conformer model.

    apply_layer_balance_to_original:
        False: the original concat branch uses raw layer outputs; the semantic branch uses LayerNorm + learnable scale.
        True : apply per-layer balancing to both branches.
    """
    model = SemanticAwareConformer(
        n_mels=n_mels,
        num_blocks=num_blocks,
        output_size=output_size,
        embedding_dim=embedding_dim,
        input_layer=input_layer,
        pos_enc_layer_type=pos_enc_layer_type,
        apply_layer_balance_to_original=apply_layer_balance_to_original,
    )
    return model


# Test function


def test_semantic_conformer():
    """Quick test for the semantic-aware model."""
    print("Testing Semantic-Aware Conformer model:")

    model = conformer_cat(num_blocks=6)

    # Test input
    x = torch.randn(2, 1, 80, 300)  # [B, 1, F, T]
    print(f"Input shape: {x.shape}")

    # Forward
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Print fusion info
    print(f"Fusion strategy:\n{model.get_fusion_info()}")

    # Parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    return model


if __name__ == "__main__":
    model = test_semantic_model()