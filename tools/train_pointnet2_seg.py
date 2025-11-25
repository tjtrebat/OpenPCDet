import numpy as np

from torch.utils.data import Dataset, DataLoader

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset


class PointSegDataset(Dataset):
    def __init__(self, kitti_dataset):
        self.kitti_dataset = kitti_dataset

    def __len__(self):
        return len(self.kitti_dataset)

    def __getitem__(self, idx):
        sample = self.kitti_dataset[idx]
        points = sample['points'][:, :4]        
        N = points.shape[0]
        sem_labels = np.zeros(N, dtype=np.int32)
        one_hot_labels = sample['points'][:, 4:7]
        foreground_mask = np.any(one_hot_labels, axis=1)
        sem_labels[foreground_mask] = np.argmax(one_hot_labels[foreground_mask], axis=1) + 1
        return points, sem_labels


cfg_from_yaml_file('tools/cfgs/dataset_configs/kitti_dataset.yaml', cfg)
cfg.DATA_PROCESSOR = [{
    'NAME': 'mask_points_and_boxes_outside_range',
    'REMOVE_OUTSIDE_BOXES': True
}]
class_names = ['Car', 'Pedestrian', 'Cyclist']
kitti_dataset = KittiDataset(cfg, class_names, training=True)

dataset = PointSegDataset(kitti_dataset)
points, labels = dataset[0]
print(points.shape, labels.shape)
print('Background Points:', len(labels[labels == 0]))
print('Car Points:', len(labels[labels == 1]))
print('Pedestrian Points:', len(labels[labels == 2]))
print('Cyclist Points:', len(labels[labels == 3]))
