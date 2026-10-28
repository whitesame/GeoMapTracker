"""Construct the selected N1 architecture for complete nuScenes oldsplit.

Dataset size changes alter the schedule, never the N1 prediction architecture.
The Stage1 recipe is a new full-data run, not a bitwise reconstruction of the
historical shared Stage1 checkpoint used during the ablation study.
"""
from copy import deepcopy
from math import ceil
import os

from .n1_template import TEMPLATE


TRAIN_FRAMES = 27968
VAL_FRAMES = 6019
TRAIN_SCENES = 696
VAL_SCENES = 150


def _positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError('%s must be a positive integer' % name)
    return value


def make_config(stage):
    """Return plain values accepted by MMCV Config, without imported modules."""
    if stage not in ('stage1', 'stage2', 'stage3', 'inference'):
        raise ValueError('Unknown stage: %s' % stage)
    schedule_stage = 'stage3' if stage == 'inference' else stage
    cfg = deepcopy(TEMPLATE)
    gpu_count = _positive_env('GEOMAP_NUM_GPUS', 2)
    defaults = {'stage1': 1, 'stage2': 17, 'stage3': 5}
    batch = _positive_env('GEOMAP_%s_BATCH' % schedule_stage.upper(), defaults[schedule_stage])
    epochs = {'stage1': 18, 'stage2': 6, 'stage3': 48}[schedule_stage]
    steps = ceil(TRAIN_FRAMES / (gpu_count * batch))
    root = os.environ.get('NUSCENES_DATA_ROOT', './datasets/nuScenes')
    annotation_root = os.environ.get('GEOMAP_INFO_ROOT', 'data_infos/nuscenes')
    output_root = os.environ.get('GEOMAP_WORK_ROOT', 'work_dirs/geomaptracker')
    work_dir = os.path.join(output_root, stage)
    annotation_paths = {
        split: os.path.join(annotation_root, 'nuscenes_map_infos_%s.pkl' % split)
        for split in ('train', 'val')
    }

    def rewrite(value):
        if isinstance(value, dict):
            if value.get('type') == 'NuscDataset':
                split = 'train' if value['ann_file'].endswith('_train.pkl') else 'val'
                value.update(data_root=root, ann_file=annotation_paths[split], interval=1)
            if 'work_dir' in value:
                value['work_dir'] = work_dir
            for child in value.values():
                rewrite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                rewrite(child)

    rewrite(cfg)
    cfg.update(
        work_dir=work_dir, num_gpus=gpu_count, batch_size=batch,
        num_iters_per_epoch=steps, num_epochs=epochs,
        num_epochs_interval={'stage1': 3, 'stage2': 1, 'stage3': 6}[schedule_stage],
        total_iters=epochs * steps, resume_from=None,
    )
    cfg['data']['samples_per_gpu'] = batch
    cfg['data']['workers_per_gpu'] = _positive_env('GEOMAP_WORKERS_PER_GPU', 4)
    cfg['runner']['max_iters'] = cfg['total_iters']
    cfg['checkpoint_config'].update(interval=steps, max_keep_ckpts=3, save_last=True)
    cfg['evaluation']['interval'] = cfg['num_epochs_interval'] * steps
    cfg['model']['history_cfg']['diagnostics_path'] = os.path.join(work_dir, 'history_diagnostics.jsonl')
    # The CUDA attention kernel requires flattened (batch * six cameras) to be
    # divisible by its effective im2col step. Preserve both source N1 defaults;
    # choose a valid divisor when a different per-GPU batch is requested.
    camera_batch = batch * 6
    if schedule_stage == 'stage2' and batch == 17:
        im2col_step = 34
    elif schedule_stage == 'stage3' and batch == 5:
        im2col_step = 30
    else:
        im2col_step = max(x for x in range(1, min(64, camera_batch) + 1)
                          if camera_batch % x == 0)
    cfg['model']['backbone_cfg']['transformer']['encoder']['transformerlayers'][
        'attn_cfgs'][1]['deformable_attention']['im2col_step'] = im2col_step
    cfg['SyncBN'] = False  # selected N1 used per-GPU BatchNorm
    cfg['seed'] = 0

    if schedule_stage == 'stage1':
        cfg['load_from'] = None
        cfg['model'].update(skip_vector_head=True, freeze_bev=False,
                            mem_warmup_iters=cfg['total_iters'])
        # BEV pretraining predicts semantic masks without vector predictions.
        # Follow the upstream Stage1 evaluator while keeping GT rasterization
        # exclusively in the evaluator's dataset, outside model inputs.
        semantic_eval = deepcopy(cfg['eval_config'])
        semantic_eval['pipeline'].insert(1, dict(
            type='RasterizeMap', roi_size=cfg['roi_size'], coords_dim=cfg['coords_dim'],
            canvas_size=cfg['canvas_size'], thickness=cfg['thickness'], semantic_mask=True))
        semantic_eval['pipeline'][-1]['keys'] = ['vectors', 'semantic_mask']
        cfg['eval_config'] = semantic_eval
        for split in ('val', 'test'):
            cfg['data'][split]['eval_semantic'] = True
            cfg['data'][split]['eval_config'] = deepcopy(semantic_eval)
        cfg['model']['backbone_cfg']['img_backbone']['pretrained'] = os.environ.get(
            'GEOMAP_IMAGENET_CHECKPOINT', 'ckpts/resnet50_msra-5891d200.pth')
        cfg['optimizer'].update(
            lr=5e-4 * gpu_count * batch / 8,
            paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
        cfg['lr_config'].update(warmup_iters=500, min_lr_ratio=0.05)
    elif schedule_stage == 'stage2':
        cfg['load_from'] = os.path.join(output_root, 'stage1', 'latest.pth')
        cfg['model'].update(skip_vector_head=False, freeze_bev=True,
                            mem_warmup_iters=ceil(71 * steps / 83))
        cfg['optimizer'].update(
            lr=0.0003541666666666667 * gpu_count * batch / 34,
            paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
        cfg['lr_config'].update(warmup_iters=ceil(9 * steps / 83), min_lr_ratio=0.95)
    else:
        cfg['load_from'] = os.path.join(output_root, 'stage2', 'latest.pth')
        cfg['model'].update(skip_vector_head=False, freeze_bev=False, mem_warmup_iters=-1)
        cfg['optimizer']['lr'] = 0.00046875 * gpu_count * batch / 10
        cfg['lr_config'].update(warmup_iters=ceil(10 * steps / 282), min_lr_ratio=0.003)

    if stage == 'inference':
        cfg['load_from'] = os.path.join(output_root, 'stage3', 'latest.pth')
    cfg['experiment'] = dict(
        variant='n1_bev_no_ce', stage=stage, dataset='nuScenes oldsplit full trainval',
        train_frames=TRAIN_FRAMES, val_frames=VAL_FRAMES,
        train_scenes=TRAIN_SCENES, val_scenes=VAL_SCENES,
        global_batch=gpu_count * batch, seed=0,
        protocol='New full-data 18/6/48 training recipe; N1 architecture; no SAT or CE',
        lr_policy='Linear global-batch scaling; warmup epoch fractions preserved from selected N1',
        initialization='Fresh ImageNet Stage1; independent optimizer at each stage',
    )
    return cfg
