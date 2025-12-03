import torch
import numpy as np

from pathlib import Path

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


def run_and_save_predictions(model, loader, device, save_dir):
    """
    Runs semantic predictions and saves:
        - points (N, C)
        - predicted labels (N,)
        - (optional) ground-truth labels (N,)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_dict in loader:
            for k, v in batch_dict.items():
                if isinstance(v, torch.Tensor):
                    batch_dict[k] = v.to(device)
            out = model(batch_dict)
            preds = out['logits'].argmax(dim=1).cpu().numpy()
            points = batch_dict['points'].cpu().numpy()
            if 'labels' in batch_dict:
                gt_labels = batch_dict['labels'].cpu().numpy()
            else:
                gt_labels = None
            frame_id = batch_dict['frame_id']
            out_file = save_dir / f"predictions_{frame_id}.npz"
            if gt_labels is not None:
                np.savez_compressed(
                    out_file,
                    points=points,
                    pred_labels=preds,
                    gt_labels=gt_labels,
                )
            else:
                np.savez_compressed(
                    out_file,
                    points=points,
                    pred_labels=preds,
                )
            print(f"Saved: {out_file}")


def main():
    device = torch.device('cuda')

    cfg_from_yaml_file('cfgs/dataset_configs/kitti_dataset.yaml', cfg)
    cfg.DATA_PROCESSOR = [
        {'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': False},
        {'NAME': 'sample_points', 'NUM_POINTS': {'train': 20000, 'test': 20000}},
        {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}},
    ]

    class_names = ['Car', 'Pedestrian', 'Cyclist']

    kitti_val = KittiDataset(cfg, class_names, training=True)
    dataset = PointSegDataset(kitti_val)

    loader = DataLoader(
        dataset,
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

    save_dir = '/projects/bfqr/ttrebat/kitti/training/seg_labels'

    run_and_save_predictions(model, loader, device, save_dir)


if __name__ == '__main__':
    main()
