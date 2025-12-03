import torch.nn as nn

from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2MSG

class PointNet2Seg(nn.Module):
    def __init__(self, model_cfg, input_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = PointNet2MSG(model_cfg, input_channels=input_channels)
        feat_dim = self.backbone.num_point_features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, batch_dict):
        batch_dict = self.backbone(batch_dict)
        point_features = batch_dict['point_features']     # (total_points, C)
        logits = self.classifier(point_features)          # (total_points, num_classes)
        batch_dict['logits'] = logits
        return batch_dict
