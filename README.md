# GeoMapTracker

基于 [MapTracker](https://github.com/woodfrog/maptracker) 的矢量建图研究代码。本仓库整理自实际完成训练的 **N1 / `n1_bev_no_ce`**，保留上游 `plugin/`、`tools/`、`docs/` 和配置组织方式。

N1 保留原生 vector memory、temporal BEV、MotionMLP 和变换辅助损失，修正跟踪传播的源位姿，并加入实例历史引导的 BEV MTP。历史采用 overwrite 更新，无类别嵌入、SAT 或退休实例回收。

历史固定 1/10 子集、seed 0、Stage3 第 48 轮的结果为 **42.5290% mAP / 34.4727% C-mAP**。这里发布的默认入口已改为**完整 nuScenes oldsplit**：27,968 帧训练、6,019 帧验证，696/150 个场景，所有数据入口 `interval=1`。历史子集精度不是全量精度；本次整理没有重新进行全量训练。来源与验证见 [N1 迁移说明](docs/n1_migration.md)。

## Docker 环境

训练依赖仅安装在 Docker 内。宿主需要 Docker、NVIDIA Container Toolkit 和可用的 NVIDIA 驱动。

```bash
docker build -f docker/Dockerfile -t geomaptracker:cu117 .
export NUSCENES_HOST_ROOT=/absolute/path/to/nuScenes
```

环境采用 Python 3.8、PyTorch 1.13.1+cu117、MMCV 1.7.0、mmdet 2.28.2、mmseg 0.30.0 和 mmdet3d 1.0.0rc6。Dockerfile 从公开依赖构建，无需服务器私有 wheel 或预先复制 mmdetection3d。

已有实验镜像的机器可直接复用：

```bash
export GEOMAP_IMAGE=histrackmap:cu117-20261012
bash docker/run.sh cpu python -c 'import plugin; print("OK")'
```

本次已在原实验镜像中验证代码；新增 Dockerfile 尚未完整重建。上游历史 Conda 安装方式保留在 [installation.md](docs/installation.md)，当前推荐使用 Docker。

## 完整数据准备

原始数据目录应包含 `samples/`、`v1.0-trainval/` 和 `maps/expansion/` 下四个区域的地图 JSON。代码使用 MapTracker oldsplit 的排除规则，故训练场景为 696 个。

```bash
# 标注写入 data_infos/nuscenes/；原始数据始终以只读方式挂载。
bash docker/run.sh cpu python tools/geomaptracker/prepare_data.py

# 在训练机器上为完整 train/val 生成跟踪 GT。
bash docker/run.sh cpu python tools/tracking/prepare_gt_tracks.py \
  plugin/configs/geomaptracker/nusc_oldsplit_stage3.py \
  --out-dir tracking_gts/nuscenes

# 校验完整帧数、场景、跟踪顺序、地图与全部相机图片。
bash docker/run.sh cpu python -m tools.geomaptracker.validate --check-images
```

需要以下四个生成文件；不接受预先截取的 1/10 标注：

```text
data_infos/nuscenes/
├── nuscenes_map_infos_train.pkl
├── nuscenes_map_infos_train_gt_tracks.pkl
├── nuscenes_map_infos_val.pkl
└── nuscenes_map_infos_val_gt_tracks.pkl
```

复用可信的完整标注时，可将其放在 `data_infos/imported_full/`，用下面的命令保留全部帧、顺序和 ID，同时改写旧图片根路径：

```bash
bash docker/run.sh cpu python tools/geomaptracker/prepare_data.py \
  --reuse-infos data_infos/imported_full \
  --source-root /old/nuScenes \
  --data-root /workspace/GeoMapTracker/datasets/nuScenes
```

脚本不会覆盖已有输出。跨机器迁移时，跟踪 GT 推荐依照上游要求重新生成。也可显式设置 `NUSCENES_CONTAINER_ALIAS=/old/nuScenes` 兼容已有绝对图片路径。若数据目录含指向目录外的软链接，用 `GEOMAP_NUSCENES_EXTRA_HOST_ROOT` 和 `GEOMAP_NUSCENES_EXTRA_CONTAINER_ROOT` 额外挂载目标，或使用完整的物理数据目录。

## 训练和评估

先将 ImageNet ResNet50 Caffe 初始化权重 `resnet50_msra-5891d200.pth` 放入 `ckpts/`，也可通过 `GEOMAP_IMAGENET_CHECKPOINT` 指定容器内路径。数据、权重与训练输出均不提交到 Git。

明确选择当前可用的 GPU 后，依次完成三个阶段：

```bash
export GEOMAP_GPUS=0,1
export GEOMAP_NUM_GPUS=2

for stage in 1 2 3; do
  bash docker/run.sh gpu bash tools/dist_train.sh \
    "plugin/configs/geomaptracker/nusc_oldsplit_stage${stage}.py" \
    "$GEOMAP_NUM_GPUS" || break
done
```

默认配置使用 2 卡，Stage2/3 的 batch 沿用 N1 在 48GB 显存卡上的设置：

| 阶段 | 每卡 batch | 轮数 | 每轮步数 | 总步数 |
|---|---:|---:|---:|---:|
| Stage1 BEV 预训练 | 1 | 18 | 13,984 | 251,712 |
| Stage2 冻结 BEV | 17 | 6 | 823 | 4,938 |
| Stage3 联合训练 | 5 | 48 | 2,797 | 134,256 |

Stage2 加载 `work_dirs/geomaptracker/stage1/latest.pth`，Stage3 加载自己完整 Stage2 的 `latest.pth`，各阶段新建优化器。新的 Stage1 从 ImageNet 开始，不等同于历史实验的共享 Stage1 权重。不要用早停或 20 步短跑权重替代完整阶段权重。

显存不足时可设置 `GEOMAP_STAGE1_BATCH`、`GEOMAP_STAGE2_BATCH`、`GEOMAP_STAGE3_BATCH`。配置按 `ceil(27968 / global_batch)` 同步更新步数、检查点、验证间隔和 warmup，学习率按全局 batch 线性缩放。其他参数包括 `GEOMAP_WORKERS_PER_GPU`、`GEOMAP_INFO_ROOT`、`GEOMAP_WORK_ROOT`。改变 batch 或卡数会改变训练轨迹。

```bash
# 检查配置；无需读取数据。
bash docker/run.sh cpu python -m tools.geomaptracker.validate --config-only

# 正式 Stage3 权重，全量验证 mAP。
export GEOMAP_GPUS=0
export GEOMAP_NUM_GPUS=1
bash docker/run.sh gpu python tools/test.py \
  plugin/configs/geomaptracker/nusc_oldsplit_inference.py \
  work_dirs/geomaptracker/stage3/latest.pth --eval

# C-mAP 使用同一 checkpoint 导出的传播预测。
bash docker/run.sh cpu python tools/tracking/calculate_cmap.py \
  plugin/configs/geomaptracker/nusc_oldsplit_inference.py \
  --result_path work_dirs/geomaptracker/inference/pos_predictions.pkl --consist 1
```

`GEOMAP_NUM_GPUS` 必须与 `GEOMAP_GPUS` 一致。推理配置默认指向正式 Stage3 权重，`tools/test.py` 的 checkpoint 位置参数决定实际加载文件。

## 验证与上游

```bash
bash docker/run.sh cpu python -m pytest -q -p no:cacheprovider tests/mainline
```

保留了历史几何、MTP 梯度、位姿传播、memory 搬移和严格 checkpoint 兼容的回归检查。上游项目介绍与论文引用保留在 [README.upstream.md](README.upstream.md)，原始使用说明见 [docs/](docs/)。

许可沿用上游 [LICENSE](LICENSE) 与 [LICENSE_GPL](LICENSE_GPL)，包括上游的非商业用途限制。请保留 MapTracker 及原始依赖的署名与引用。本工程是研究实现，不将已有 HisTrackMap 组件宣称为新的方法贡献。
