import torch
import numpy as np

from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.pointnet2_seg import PointNet2Seg
from pcdet.datasets.point_seg_dataset import PointSegDataset


def collate_fn(batch):
    batch_points = []
    batch_labels = []

    for b_idx, (pts, lbls) in enumerate(batch):
        N = pts.shape[0]
        batch_col = torch.full((N, 1), b_idx, dtype=torch.float32)
        batch_points.append(torch.cat([batch_col, torch.from_numpy(pts)], dim=1))
        batch_labels.append(torch.from_numpy(lbls))

    return {
        'batch_size': len(batch),
        'points': torch.cat(batch_points, dim=0),
        'labels': torch.cat(batch_labels, dim=0),
    }


def evaluate(model, loader, device, num_classes):
    model.eval()

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    with torch.no_grad():
        for batch_dict in loader:
            for k in batch_dict:
                if isinstance(batch_dict[k], torch.Tensor):
                    batch_dict[k] = batch_dict[k].to(device)

            out = model(batch_dict)
            preds = out['logits'].argmax(dim=1).cpu().numpy()
            labels = batch_dict['labels'].cpu().numpy()

            for c in range(num_classes):
                tp[c] += np.sum((preds == c) & (labels == c))
                fp[c] += np.sum((preds == c) & (labels != c))
                fn[c] += np.sum((preds != c) & (labels == c))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall


def main():
    device = torch.device('cuda')

    cfg_from_yaml_file('cfgs/dataset_configs/kitti_dataset.yaml', cfg)
    cfg.DATA_PROCESSOR = [
        {'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': False},
        {'NAME': 'sample_points', 'NUM_POINTS': {'train': 20000, 'test': 20000}},
        {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}},
    ]

    class_names = ['Car', 'Pedestrian', 'Cyclist']

    kitti_val = KittiDataset(cfg, class_names, training=False)
    val_set = PointSegDataset(kitti_val)

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model_cfg = cfg.SEMANTIC_SEGMENTATION.MODEL_CFG

    num_classes = 4
    input_channels = 4

    model = PointNet2Seg(model_cfg, input_channels, num_classes).to(device)

    ckpt = cfg.SEMANTIC_SEGMENTATION.CKPT_PATH
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model_state'])

    precision, recall = evaluate(model, val_loader, device, num_classes)

    for c in range(num_classes):
        print(f"Class {c}: Precision={precision[c]:.4f}, Recall={recall[c]:.4f}")


if __name__ == '__main__':
    main()
