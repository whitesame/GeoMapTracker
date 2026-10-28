"""Prepare complete nuScenes oldsplit metadata without writing to the raw data.

Run inside Docker. Existing pickles must be trusted local research artifacts.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import sys


COUNTS = {'train': (27968, 696), 'val': (6019, 150)}


def check_split(samples, split):
    expected = COUNTS[split]
    actual = len(samples), len({sample['scene_name'] for sample in samples})
    if actual != expected:
        raise ValueError('%s is not full oldsplit: %s != %s' % (split, actual, expected))
    if len({sample['token'] for sample in samples}) != len(samples):
        raise ValueError('Duplicate sample tokens: ' + split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.getenv('NUSCENES_DATA_ROOT', 'datasets/nuScenes'))
    parser.add_argument('--out-dir', default=os.getenv('GEOMAP_INFO_ROOT', 'data_infos/nuscenes'))
    parser.add_argument('--reuse-infos', type=Path,
                        help='Import full annotations and matching files from this trusted directory')
    parser.add_argument('--source-root',
                        help='Old image root in reused metadata; remap its camera paths to --data-root')
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in COUNTS:
        for suffix in ('.pkl', '_gt_tracks.pkl'):
            if (output / ('nuscenes_map_infos_' + split + suffix)).exists():
                raise FileExistsError('Refusing to overwrite existing annotations in ' + str(output))
    if args.reuse_infos:
        prepared = {}
        for split in COUNTS:
            name = 'nuscenes_map_infos_' + split + '.pkl'
            with (args.reuse_infos / name).open('rb') as stream:
                samples = pickle.load(stream)
            check_split(samples, split)
            if args.source_root:
                prefix = args.source_root.rstrip('/') + '/'
                for sample in samples:
                    for camera in sample['cams'].values():
                        old = camera['img_fpath']
                        if not old.startswith(prefix):
                            raise ValueError('Camera path does not start with --source-root: ' + old)
                        camera['img_fpath'] = str(Path(args.data_root) / old[len(prefix):])
            prepared[split] = samples
        # Validate both splits before writing either one. Preserve order and IDs.
        for split, samples in prepared.items():
            name = 'nuscenes_map_infos_' + split
            with (output / (name + '.pkl')).open('wb') as stream:
                pickle.dump(samples, stream, protocol=pickle.HIGHEST_PROTOCOL)
            matching = args.reuse_infos / (name + '_gt_tracks.pkl')
            if matching.exists():
                shutil.copy2(matching, output / matching.name)
    else:
        if args.source_root:
            parser.error('--source-root requires --reuse-infos')
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data_converter'))
        from nuscenes_converter import create_nuscenes_infos_map
        create_nuscenes_infos_map(root_path=args.data_root, dest_path=str(output),
                                  version='v1.0-trainval', new_split=False)
        for split in COUNTS:
            with (output / ('nuscenes_map_infos_' + split + '.pkl')).open('rb') as stream:
                check_split(pickle.load(stream), split)
    print(json.dumps({'annotations': str(output), 'full_oldsplit': COUNTS,
                      'next': 'Generate/check train and val _gt_tracks.pkl on the training machine.'}))


if __name__ == '__main__':
    main()
