# N1 来源与完整数据迁移

整理日期：2026-10-28。

## 版本选择

来源为服务器 `/data/cy/histrackmap_adaptive_20261021/project/` 的实际 N1 实验，正式配置为 `work_dirs/adaptive/configs/n1_bev_no_ce_stage3.py`。选择依据为总实验记录与该实验归档的 `final_metrics.json`，使用 Stage3 第 48 轮的同一 checkpoint 配对指标。

| 历史 N1 项目 | 记录 |
|---|---|
| 训练/验证 | 固定 1/10，2815/603 帧，70/15 场景 |
| seed | 0 |
| Stage3 最终步数 | 13,536 |
| mAP | 42.5290% |
| C-mAP | 34.4727280882% |
| 正式权重 SHA256 | `61ae0dc122805eb9034b84dacd6b492b0e66d9af225845f0263e8c1e6cb14712` |

模型保留原生 VEC memory、temporal BEV、MotionMLP、变换辅助损失和原始阈值，传播使用实际源位姿。显式历史为 overwrite + BEV MTP，无 CE、SAT 或退休实例注入。没有合入 RMU、decay、max、PV 读取或 readout 缩放等后续消融。

45 个模型 Python 源文件与所选 N1 原版逐字节一致。[n1_origin.json](n1_origin.json) 保存源文件哈希、原正式配置哈希与历史权重哈希。数据集与通用训练工具同样从 N1 项目导入；只移除了旧子集配置入口和服务器实验队列、监控、备份等非发布文件。

原生成的 `n1_bev_no_ce_inference.py` 指向 20 步短跑权重，不适合作为正式模型入口。新推理配置改为加载 `work_dirs/geomaptracker/stage3/latest.pth`。

## 完整数据配置

当前四个入口位于 `plugin/configs/geomaptracker/`，由 `tools/geomaptracker/config.py` 构造。所有 train/val/test、eval_config、match_config 及嵌套评估配置均使用完整标注，`interval=1`。

实际完整 oldsplit 标注包含 27,968 帧/696 训练场景及 6,019 帧/150 验证场景。跟踪 GT 必须与完整标注的场景、样本 ID 及顺序相符，不能与 1/10 标注混用。

默认两卡配置按 `ceil(训练帧数 / 全局 batch)` 重算调度：Stage1 为 251,712 步，Stage2 为 4,938 步，Stage3 为 134,256 步。检查点、验证和 memory/LR warmup 同步更新；Stage2/3 默认 batch 与学习率保持所选 N1 的设置。Stage1 验证使用分割 GT 和语义评估，后续阶段使用矢量评估。评估 GT 始终独立于模型图片输入。

新的 Stage1 是在相同 N1 参数树上从 ImageNet 开始的 18 轮完整数据 BEV 预训练方案。历史 N1 复用了另一实现产生并转换的共享 Stage1，因此新训练不是对历史初始化或随机轨迹的逐位复现。本仓库没有报告全量训练精度；历史子集数字不可移作全量数字。

## 已执行验证

验证使用原实验的 Docker 镜像 `histrackmap:cu117-20261012`，镜像 ID 为 `sha256:c909c585c2a32325ff428dbf20bfd5b567ddf336b292aa2923470bb251d358da`。

- 60 项既有模型回归测试通过，涵盖历史几何、MTP 梯度、位姿传播、memory 搬移、RNG 和 checkpoint 兼容。测试夹具改为标准上游配置并禁用测试中的在线预训练下载。
- 四份配置经 MMCV 加载和序列化检查；完整 Stage3 模型严格加载历史 N1 正式权重。742 个持久状态张量通过键、形状、dtype、有限值检查；只按目标模型声明验证并移除 MMCV 额外保存的两个非持久 buffer。
- 实际构建完整训练和验证数据集，并分别成功取样。
- 完整训练/验证的 203,922 张相机图片全部存在，完整跟踪 GT 的场景及样本顺序匹配，四张区域地图均存在。
- Stage1 的真实评估 GT 生成了 `3×100×200` 分割 mask，BEV 预训练预测的格式化导出通过。
- Docker 启动脚本经服务器 `bash -n` 检查。

以上是 CPU、数据与权重兼容检查。本次未启动新的 GPU 训练，未重新测量全量 mAP/C-mAP，新增公开 Dockerfile 尚未完成全新构建。

## 发布范围与上游

仓库包含源码、完整数据配置、Docker 配方、验证工具、必要资源及文档。原始数据、模型权重、生成标注、训练输出和实验记录不纳入 Git。服务器原实验目录及其冻结文件不修改；用于发布的副本单独保存。

上游介绍及引用保留在 [README.upstream.md](../README.upstream.md)，许可原样保留为 [LICENSE](../LICENSE) 和 [LICENSE_GPL](../LICENSE_GPL)。
