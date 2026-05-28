import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """Circle Loss for deep feature learning"""

    def __init__(self, embedding_dim=256, num_classes=1000, scale=256, margin=0.25, mode='class'):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.mode = mode

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Circle Loss parameters
        self.op = 1 + margin  # O_p = 1 + m
        self.on = -margin  # O_n = -m
        self.deltap = 1 - margin  # Δ_p = 1 - m
        self.deltan = margin  # Δ_n = m

    def forward(self, feats, labels):
        """
        Args:
            feats: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size,)
        """
        batch_size = feats.size(0)

        # Normalize features
        feats_norm = F.normalize(feats, p=2, dim=1)

        # Cosine similarity matrix
        sim_mat = torch.matmul(feats_norm, feats_norm.t())

        # Label match matrix
        label_mat = labels.unsqueeze(1) == labels.unsqueeze(0)

        # Separate positive/negative pairs
        pos_mask = label_mat.float() - torch.eye(batch_size, device=feats.device)
        neg_mask = (1 - label_mat.float())

        # Positive/negative similarities
        sp = sim_mat * pos_mask
        sn = sim_mat * neg_mask

        # Circle Loss computation
        ap = torch.clamp(self.op - sp.detach(), min=0.0)
        an = torch.clamp(sn.detach() - self.on, min=0.0)

        # Reweighted similarities
        reweight_sp = ap * (sp - self.deltap)
        reweight_sn = an * (sn - self.deltan)

        # Prevent numeric overflow
        reweight_sp = torch.clamp(reweight_sp, max=10.0)
        reweight_sn = torch.clamp(reweight_sn, max=10.0)

        # Loss
        pos_term = torch.logsumexp(self.scale * reweight_sn, dim=1)
        neg_term = torch.logsumexp(self.scale * reweight_sp, dim=1)

        loss = F.softplus(pos_term + neg_term).mean()

        # Accuracy (using the classifier)
        logits = self.classifier(feats)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == labels).float().mean()

        return loss, acc


class MultiTaskCircleLoss(nn.Module):
    """Multi-task Circle Loss combining Circle Loss and Cross-Entropy."""

    def __init__(self, embedding_dim=256, num_classes=1000, scale=256, margin=0.25,
                 lambda_circle=1.0, lambda_ce=0.5):
        super(MultiTaskCircleLoss, self).__init__()
        self.lambda_circle = lambda_circle
        self.lambda_ce = lambda_ce

        # Circle Loss
        self.circle_loss = CircleLoss(embedding_dim, num_classes, scale, margin)

        # Cross-Entropy loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, feats, labels):
        # Circle Loss
        circle_loss, circle_acc = self.circle_loss(feats, labels)

        # Cross-Entropy loss
        logits = self.circle_loss.classifier(feats)
        ce_loss = self.ce_loss(logits, labels)

        # Accuracy
        pred = torch.argmax(logits, dim=1)
        acc = (pred == labels).float().mean()

        # Combined loss
        total_loss = self.lambda_circle * circle_loss + self.lambda_ce * ce_loss

        return total_loss, acc


# Simplified Circle Loss (compatible with the existing loss module interface)
class SimpleCircleLoss(nn.Module):
    """A simplified Circle Loss that matches the existing loss-module interface."""

    def __init__(self, embedding_dim=256, num_classes=1000, scale=256, margin=0.25):
        super(SimpleCircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, feats, labels):
        batch_size = feats.size(0)

        # Normalize features
        feats_norm = F.normalize(feats, p=2, dim=1)

        # Cosine similarity matrix
        sim_mat = torch.matmul(feats_norm, feats_norm.t())

        # Label match matrix
        label_mat = labels.unsqueeze(1) == labels.unsqueeze(0)

        # Separate positive/negative pairs
        pos_mask = label_mat.float() - torch.eye(batch_size, device=feats.device)
        neg_mask = (1 - label_mat.float())

        # Circle Loss parameters
        op = 1 + self.margin
        on = -self.margin
        deltap = 1 - self.margin
        deltan = self.margin

        # Positive/negative similarities
        sp = sim_mat * pos_mask
        sn = sim_mat * neg_mask

        # Adaptive weights
        ap = torch.clamp(op - sp.detach(), min=0.0)
        an = torch.clamp(sn.detach() - on, min=0.0)

        # Reweighted similarities
        reweight_sp = ap * (sp - deltap)
        reweight_sn = an * (sn - deltan)

        # Loss
        pos_term = torch.logsumexp(self.scale * reweight_sn, dim=1)
        neg_term = torch.logsumexp(self.scale * reweight_sp, dim=1)

        loss = F.softplus(pos_term + neg_term).mean()

        # Accuracy
        logits = self.classifier(feats)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == labels).float().mean()

        return loss, acc