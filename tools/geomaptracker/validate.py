"""Check full-data annotations, temporal matching and N1 training schedules.

Run from the repository root. No training dependencies are required beyond those
needed to deserialize the existing annotation pickle files.
"""
import argparse
import json
from pathlib import Path
import pickle
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.geomaptracker.config import (  # noqa: E402
    TRAIN_FRAMES, TRAIN_SCENES, VAL_FRAMES, VAL_SCENES,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(config):
    """Reject inconsistent schedules and any remaining annotation subsampling."""
    cfg = {k: v for k, v in runpy.run_path(str(config)).items() if not k.startswith('__')}
    model = cfg['model']
    require(model['type'] == 'HisTrackMapMainline', 'Expected selected N1 mapper')
    for key in ('prior_enabled', 'use_memory', 'use_motion_mlp', 'transform_loss_enabled'):
        require(model[key] is True, 'N1 requires model.%s=True' % key)
    require(model['track_transform_source'] == 'state_pose', 'N1 requires source-pose correction')
    require(model['rmu_cfg']['mode'] == 'overwrite', 'N1 requires overwrite history updates')
    require(model['mtp_cfg']['mode'] == 'bev' and model['mtp_cfg']['use_category'] is False,
            'N1 requires BEV MTP without category embeddings')
    require(model['retired_enabled'] is False, 'N1 does not use retired priors')
    require(cfg['data']['train']['matching'] is True, 'Temporal matching must be enabled')
    expected_steps = (TRAIN_FRAMES + cfg['num_gpus'] * cfg['batch_size'] - 1) // (
        cfg['num_gpus'] * cfg['batch_size'])
    require(cfg['num_iters_per_epoch'] == expected_steps, 'Training schedule does not cover full split')
    require(cfg['total_iters'] == expected_steps * cfg['num_epochs'], 'Incorrect total iterations')
    require(cfg['runner']['max_iters'] == cfg['total_iters'], 'Runner schedule differs')
    require(cfg['evaluation']['interval'] == expected_steps * cfg['num_epochs_interval'],
            'Evaluation interval differs from epoch schedule')
    require(cfg['checkpoint_config']['interval'] == expected_steps, 'Checkpoint interval differs')
    require(cfg['data']['samples_per_gpu'] == cfg['batch_size'], 'Per-GPU batch differs')
    stage1 = cfg['experiment']['stage'] == 'stage1'
    require(model['skip_vector_head'] is stage1, 'Vector-head setting differs from training stage')
    for split in ('val', 'test'):
        dataset = cfg['data'][split]
        require(dataset.get('eval_semantic', False) is stage1,
                'Stage1 must evaluate semantic masks; later stages must evaluate vectors')
        require(dataset['pipeline'][-1]['keys'] == ['img'], 'Evaluation model inputs must contain only images')
        require(not any(item['type'] in ('VectorizeMap', 'RasterizeMap') for item in dataset['pipeline']),
                'GT geometry must stay outside the evaluation model-input pipeline')
        if stage1:
            pipeline = dataset['eval_config']['pipeline']
            require(any(item['type'] == 'RasterizeMap' and item.get('semantic_mask') is True
                        for item in pipeline), 'Stage1 evaluator requires rasterized semantic GT')
            require(set(pipeline[-1]['keys']) == {'vectors', 'semantic_mask'},
                    'Stage1 evaluator must collect semantic masks')
    train = cfg['data']['train']
    val = cfg['data']['val']
    annotation_paths = {train['ann_file'], val['ann_file']}
    require(Path(train['ann_file']).name == 'nuscenes_map_infos_train.pkl',
            'Unexpected train annotation name')
    require(Path(val['ann_file']).name == 'nuscenes_map_infos_val.pkl',
            'Unexpected validation annotation name')
    dataset_count = 0

    def visit(value):
        nonlocal dataset_count
        if isinstance(value, dict):
            if value.get('type') == 'NuscDataset':
                dataset_count += 1
                require(value.get('interval', 1) == 1, 'Dataset interval must be 1')
                require(value['ann_file'] in annotation_paths, 'Nested dataset uses different annotations')
                require(value['data_root'] == train['data_root'], 'Nested dataset uses another data root')
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(cfg)
    require(dataset_count >= 7, 'Missing full-data eval/match/train/val/test configurations')
    return cfg


def check_annotations(path, expected_frames, expected_scenes, all_images=False):
    path = Path(path)
    with path.open('rb') as handle:
        annotations = pickle.load(handle)
    require(isinstance(annotations, list), 'Annotations must be a list: %s' % path)
    require(len(annotations) == expected_frames,
            '%s has %s frames; expected full oldsplit %s (a subset is not accepted)' %
            (path, len(annotations), expected_frames))
    tokens = [item['token'] for item in annotations]
    require(len(set(tokens)) == expected_frames, 'Duplicate frame tokens: %s' % path)
    scenes = {}
    for index, item in enumerate(annotations):
        scenes.setdefault(item['scene_name'], []).append(index)
        require(len(item['cams']) == 6, 'Expected six camera views at index %s' % index)
    require(len(scenes) == expected_scenes,
            '%s has %s scenes; expected %s' % (path, len(scenes), expected_scenes))
    matching_path = path.with_name(path.stem + '_gt_tracks.pkl')
    with matching_path.open('rb') as handle:
        matching = pickle.load(handle)
    require(set(matching) == set(scenes), 'Matching scene set differs: %s' % matching_path)
    for scene, ids in scenes.items():
        info = matching[scene]
        require([int(x) for x in info['sample_ids']] == ids,
                'Matching sample IDs/order differ in scene %s' % scene)
        require(len(info['instance_ids']) == len(ids), 'Missing matching instances in scene %s' % scene)
    indices = range(len(annotations)) if all_images else sorted({ids[0] for ids in scenes.values()})
    checked_images = 0
    for index in indices:
        for camera, info in annotations[index]['cams'].items():
            image_path = Path(info['img_fpath'])
            require(image_path.is_file(), 'Missing camera image at frame %s %s: %s' %
                    (index, camera, image_path))
            checked_images += 1
    return dict(frames=len(annotations), scenes=len(scenes), matching='aligned',
                checked_images=checked_images,
                image_check='all' if all_images else 'all six views of the first frame of every scene')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='plugin/configs/geomaptracker/nusc_oldsplit_stage3.py')
    parser.add_argument('--config-only', action='store_true', help='Check schedules without reading data files')
    parser.add_argument('--check-images', action='store_true', help='Check every camera image, not scene samples')
    args = parser.parse_args()
    cfg = check_config(args.config)
    report = dict(config=args.config, stage=cfg['experiment']['stage'],
                  iterations_per_epoch=cfg['num_iters_per_epoch'], max_iters=cfg['total_iters'],
                  global_batch=cfg['experiment']['global_batch'], learning_rate=cfg['optimizer']['lr'])
    if not args.config_only:
        report['train'] = check_annotations(cfg['data']['train']['ann_file'],
                                            TRAIN_FRAMES, TRAIN_SCENES, args.check_images)
        report['val'] = check_annotations(cfg['data']['val']['ann_file'],
                                          VAL_FRAMES, VAL_SCENES, args.check_images)
        map_root = Path(cfg['data']['train']['data_root']) / 'maps' / 'expansion'
        for location in ('boston-seaport', 'singapore-hollandvillage',
                         'singapore-onenorth', 'singapore-queenstown'):
            require((map_root / (location + '.json')).is_file(), 'Missing nuScenes map: %s' % location)
        report['maps'] = 'all four expansion maps present'
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
