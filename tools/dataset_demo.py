from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

cfg_from_yaml_file('tools/cfgs/kitti_models/pointrcnn.yaml', cfg)
dataset = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False)

sample = dataset[0]
print(sample['points'].shape)
