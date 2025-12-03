import pickle
import time

import numpy as np
import torch
import tqdm

from sklearn.cluster import DBSCAN

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils



def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None, seg_model=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        if seg_model is not None:
            N = batch_dict['points'].shape[0]
            sem_labels = np.zeros(N, dtype=np.int32)
            inst_labels = np.zeros(N, dtype=np.int32)
            for batch_idx in np.unique(batch_dict['points'][:, 0]):
                mask = batch_dict['points'][:, 0] == batch_idx
                points = batch_dict['points'][mask, 1:5]
                sem_labels[mask], inst_labels[mask] = predict_semantic_and_instance(seg_model, points)
            num_classes = seg_model.num_classes
            one_hot_sem = np.zeros((N, num_classes), dtype=np.float32)
            valid = (sem_labels > 0) & (sem_labels <= num_classes)
            one_hot_sem[valid, sem_labels[valid]] = 1.0
            one_hot_sem[~valid, 0] = 1.0
            batch_dict['points'] = np.concatenate([batch_dict['points'][:, :5], one_hot_sem, inst_labels[:, None]], axis=1)        
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def predict_semantic_and_instance(seg_model, points, eps=0.5, min_samples=5):
    """
    points: (N, 3+C) numpy array
    pointnet2_model: trained PointNet2Seg
    Returns:
        sem_labels:  (N,) int32
        inst_labels: (N,) int32, 0 = background
    """
    
    with torch.no_grad():
        batch_dict = {
            'batch_size': 1,
            'points': torch.from_numpy(np.concatenate([np.zeros((points.shape[0], 1)), points], axis=1)).float().cuda()
        }
        out = seg_model(batch_dict)
        logits = out['logits']  # (N, num_classes)
        sem_labels = logits.argmax(dim=1).cpu().numpy()

    inst_labels = np.zeros_like(sem_labels, dtype=np.int32)
    next_inst_id = 1
    for cls_id in np.unique(sem_labels):
        if cls_id == 0:
            continue
        mask = sem_labels == cls_id
        pts_cls = points[mask, :3]
        if pts_cls.shape[0] == 0:
            continue
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_cls)
        labels = clustering.labels_
        labels[labels >= 0] += next_inst_id
        inst_labels[mask] = labels
        next_inst_id += labels.max() + 1

    return sem_labels, inst_labels

if __name__ == '__main__':
    pass
