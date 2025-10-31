import os
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

cfg_from_yaml_file('tools/cfgs/kitti_models/pointrcnn.yaml', cfg)
dataset = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=True)

sample = dataset[0]
print(sample['points'].shape)
print("Sample keys:", sample.keys())

# save_path = "/projects/bfqr/ttrebat/samples"
# os.makedirs(save_path, exist_ok=True)
# with open(os.path.join(save_path, f"sample_{sample['frame_id']}.pkl"), "wb") as f:
#     pickle.dump(sample, f)
