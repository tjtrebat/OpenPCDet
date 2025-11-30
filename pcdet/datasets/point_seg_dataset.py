import numpy as np

from torch.utils.data import Dataset


class PointSegDataset(Dataset):
    def __init__(self, kitti_dataset):
        self.kitti_dataset = kitti_dataset

    def __len__(self):
        return len(self.kitti_dataset)

    def __getitem__(self, idx):
        sample = self.kitti_dataset[idx]
        points = sample['points'][:, :4]
        N = points.shape[0]

        labels = np.zeros(N, dtype=np.int32)
        one_hot_labels = sample['points'][:, 4:8]
        labels = np.argmax(one_hot_labels, axis=1)

        return points.astype('float32'), labels.astype('int64')
