import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models.pointnet2_seg import PointNet2Seg
from pcdet.datasets.point_seg_dataset import PointSegDataset


def init_distributed():
    if "RANK" not in os.environ:
        print("Running in single-GPU mode.")
        return None, torch.device("cuda")

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return dist.get_rank(), device


def collate_fn(batch):
    batch_points = []
    batch_labels = []
    for b_idx, (pts, lbls) in enumerate(batch):
        N = pts.shape[0]
        batch_col = torch.full((N,1), b_idx, dtype=torch.float32)
        batch_points.append(torch.cat([batch_col, torch.from_numpy(pts)], dim=1))
        batch_labels.append(torch.from_numpy(lbls))

    return {
        'batch_size': len(batch),
        'points': torch.cat(batch_points, dim=0),
        'labels': torch.cat(batch_labels, dim=0),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_dict in loader:
        for k in batch_dict:
            if isinstance(batch_dict[k], torch.Tensor):
                batch_dict[k] = batch_dict[k].to(device)

        out_dict = model(batch_dict)
        logits = out_dict['logits']        # (M, num_classes)
        labels = batch_dict['labels']      # (M,)

        loss = nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    rank, device = init_distributed()
    is_main = (rank is None) or (rank == 0)

    cfg_from_yaml_file('cfgs/dataset_configs/kitti_dataset.yaml', cfg)
    cfg.DATA_PROCESSOR = [
        {'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': False}, 
        {'NAME': 'sample_points', 'NUM_POINTS': {'train': 20000, 'test': 20000}}, 
        {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}
    ]

    class_names = ['Car', 'Pedestrian', 'Cyclist']
    kitti_dataset = KittiDataset(cfg, class_names, training=True)
    train_set = PointSegDataset(kitti_dataset)

    if rank is not None:
        train_sampler = DistributedSampler(train_set, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        collate_fn=collate_fn,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model_cfg = cfg.SEMANTIC_SEGMENTATION.MODEL_CFG

    num_classes = 4
    input_channels = 3 + 1  # x,y,z + reflectance

    model = PointNet2Seg(model_cfg, input_channels, num_classes).to(device)

    if rank is not None:
        model = DDP(model, device_ids=[device], output_device=device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    max_epochs = 50
    ckpt_path = Path('/u/ttrebat/OpenPCDet/output/kitti_models/pointnet2_seg/ckpt')
    ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        if rank is not None:
            train_loader.sampler.set_epoch(epoch)
                
        loss = train_one_epoch(model, train_loader, optimizer, device)

        if is_main:
            print(f"[Epoch {epoch:03d}] Loss: {loss:.4f}")
            save_path = ckpt_path / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict() if rank is not None else model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, save_path)
            print(f"  → saved checkpoint at {save_path}")

    if rank is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()