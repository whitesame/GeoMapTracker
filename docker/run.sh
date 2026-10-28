#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash docker/run.sh cpu|gpu COMMAND [ARG ...]

CPU is suitable for data preparation, validation and CPU tests.
GPU mode requires GEOMAP_GPUS with explicit GPU indices or UUIDs, e.g. 0,1.

  NUSCENES_HOST_ROOT=/path/to/nuScenes bash docker/run.sh cpu python ...
  GEOMAP_GPUS=0,1 NUSCENES_HOST_ROOT=/path/to/nuScenes \
    bash docker/run.sh gpu bash tools/dist_train.sh CONFIG 2

Environment:
  GEOMAP_IMAGE                 image name (default: geomaptracker:cu117)
  NUSCENES_HOST_ROOT            complete nuScenes directory on the host
  GEOMAP_DATA_INFOS_HOST_ROOT   annotations host directory (default: ./data_infos)
  GEOMAP_CKPTS_HOST_ROOT        checkpoints host directory (default: ./ckpts)
  GEOMAP_WORK_DIR_HOST_ROOT     output host directory (default: ./work_dirs)
  NUSCENES_CONTAINER_ALIAS      optional absolute legacy path(s), comma-separated
  GEOMAP_NUSCENES_EXTRA_HOST_ROOT       optional symlink target on the host
  GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT  same target path inside the container
  GEOMAP_SHM_SIZE               shared memory size (default: 64g)

Legacy aliases are opt-in and expose the SAME dataset read-only at each path.
Regenerate annotations when their absolute image paths refer to another layout.
EOF
}

[[ $# -ge 2 ]] || { usage >&2; exit 2; }
mode=$1
shift
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
container_project=/workspace/GeoMapTracker
image=${GEOMAP_IMAGE:-geomaptracker:cu117}
data_root=${NUSCENES_HOST_ROOT:-$project/datasets/nuScenes}
infos_root=${GEOMAP_DATA_INFOS_HOST_ROOT:-$project/data_infos}
ckpts_root=${GEOMAP_CKPTS_HOST_ROOT:-$project/ckpts}
work_root=${GEOMAP_WORK_DIR_HOST_ROOT:-$project/work_dirs}

[[ -d "$data_root" ]] || { echo "nuScenes directory does not exist: $data_root" >&2; exit 2; }
data_root=$(cd -- "$data_root" && pwd -P)
mkdir -p -- "$infos_root" "$ckpts_root" "$work_root"
infos_root=$(cd -- "$infos_root" && pwd -P)
ckpts_root=$(cd -- "$ckpts_root" && pwd -P)
work_root=$(cd -- "$work_root" && pwd -P)

devices=()
case "$mode" in
    cpu)
        devices+=(-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void)
        ;;
    gpu)
        [[ -n ${GEOMAP_GPUS:-} && ${GEOMAP_GPUS} != all ]] || {
            echo 'Set GEOMAP_GPUS to explicit GPU indices or UUIDs (e.g. 0,1).' >&2
            exit 2
        }
        [[ "$GEOMAP_GPUS" =~ ^[A-Za-z0-9_,.:-]+$ ]] || { echo 'Invalid GEOMAP_GPUS.' >&2; exit 2; }
        [[ "$GEOMAP_GPUS" != ,* && "$GEOMAP_GPUS" != *, && "$GEOMAP_GPUS" != *,,* ]] || {
            echo 'GEOMAP_GPUS contains an empty entry.' >&2; exit 2
        }
        IFS=',' read -r -a selected_gpus <<< "$GEOMAP_GPUS"
        declare -A seen_gpus=()
        for device in "${selected_gpus[@]}"; do
            [[ -n "$device" ]] || { echo 'Empty entry in GEOMAP_GPUS.' >&2; exit 2; }
            [[ -z ${seen_gpus[$device]:-} ]] || { echo 'Duplicate entry in GEOMAP_GPUS.' >&2; exit 2; }
            seen_gpus[$device]=1
        done
        selected_count=${#selected_gpus[@]}
        [[ ${GEOMAP_NUM_GPUS:-$selected_count} == "$selected_count" ]] || {
            echo 'GEOMAP_NUM_GPUS must match the number of GEOMAP_GPUS entries.' >&2
            exit 2
        }
        export GEOMAP_NUM_GPUS=$selected_count
        devices+=(--gpus "\"device=$GEOMAP_GPUS\"")
        ;;
    *) usage >&2; exit 2 ;;
esac

args=(--rm --init --shm-size "${GEOMAP_SHM_SIZE:-64g}"
      --user "$(id -u):$(id -g)" -e HOME=/tmp
      -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1
      -e "PYTHONPATH=$container_project:$container_project/tools"
      -e "NUSCENES_DATA_ROOT=$container_project/datasets/nuScenes"
      --mount "type=bind,src=$project,dst=$container_project"
      --mount "type=bind,src=$data_root,dst=$container_project/datasets/nuScenes,readonly"
      --mount "type=bind,src=$infos_root,dst=$container_project/data_infos"
      --mount "type=bind,src=$ckpts_root,dst=$container_project/ckpts,readonly"
      --mount "type=bind,src=$work_root,dst=$container_project/work_dirs"
      --workdir "$container_project")

if [[ -n ${NUSCENES_CONTAINER_ALIAS:-} ]]; then
    IFS=',' read -r -a aliases <<< "$NUSCENES_CONTAINER_ALIAS"
    for alias_path in "${aliases[@]}"; do
        case "$alias_path" in
            /data/*|/data1/*|/mnt/*|/datasets/*) ;;
            *) echo "Legacy data alias must be under /data, /data1, /mnt or /datasets: $alias_path" >&2; exit 2 ;;
        esac
        [[ "$alias_path" != *'..'* ]] || { echo 'Legacy alias cannot contain ..' >&2; exit 2; }
        args+=(--mount "type=bind,src=$data_root,dst=$alias_path,readonly")
    done
fi

# Some existing dataset views contain absolute symlinks. Expose their backing
# directory only when explicitly requested; prefer a complete physical data root.
if [[ -n ${GEOMAP_NUSCENES_EXTRA_HOST_ROOT:-} || -n ${GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT:-} ]]; then
    [[ -n ${GEOMAP_NUSCENES_EXTRA_HOST_ROOT:-} && -n ${GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT:-} ]] || {
        echo 'Set both GEOMAP_NUSCENES_EXTRA_HOST_ROOT and GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT.' >&2
        exit 2
    }
    [[ -d "$GEOMAP_NUSCENES_EXTRA_HOST_ROOT" ]] || { echo 'Extra dataset directory does not exist.' >&2; exit 2; }
    extra_root=$(cd -- "$GEOMAP_NUSCENES_EXTRA_HOST_ROOT" && pwd -P)
    case "$GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT" in
        /data/*|/data1/*|/mnt/*|/datasets/*) ;;
        *) echo 'Extra dataset destination must be under /data, /data1, /mnt or /datasets.' >&2; exit 2 ;;
    esac
    [[ "$GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT" != *'..'* ]] || { echo 'Extra destination cannot contain ..' >&2; exit 2; }
    args+=(--mount "type=bind,src=$extra_root,dst=$GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT,readonly")
fi

# Only forward the documented configuration family, never arbitrary host secrets.
while IFS= read -r variable; do
    case "$variable" in
        GEOMAP_NUM_GPUS|GEOMAP_STAGE1_BATCH|GEOMAP_STAGE2_BATCH|GEOMAP_STAGE3_BATCH|GEOMAP_WORKERS_PER_GPU|GEOMAP_WORK_ROOT|GEOMAP_INFO_ROOT|GEOMAP_IMAGENET_CHECKPOINT)
            args+=(-e "$variable=${!variable}") ;;
    esac
done < <(compgen -e)

if [[ -t 0 && -t 1 ]]; then args+=(-it); fi
exec docker run "${args[@]}" "${devices[@]}" "$image" "$@"
