# keidroid Ogerpon Front-100 Search

该流程固定 keidroid 的60张 Ogerpon 牌组，只搜索决策策略。默认不会提交 Kaggle，也不会消耗线上额度。

## 已解决的问题

- 训练、验证和盲测按完整 episode 切分，禁止同局动作跨 split；
- artifact v2 同时保存 global、context head、outcome-weighted value 模型和置信度门槛；
- runtime 按可观测 context 路由模型，并只在低置信度单选动作上运行短搜索；
- Ogerpon runtime 记录本回合主要攻击手和能量规划特征；
- 行为审计只读取 artifact 锁定的 test episodes；
- 最终 holdout 使用严格 Kaggle profile；
- 构建包默认只保留 Kaggle Linux 的 `libcg.so`，模型保存在压缩 `model.json.gz`；
- 每次运行输出固定 deck hash、模型/包哈希和 `submission_queue.json`。
- Majkel 教师运行严格要求玩家名与固定牌组同时匹配，防止把换牌组对局或使用相同牌组的对手混入训练。
- CatBoost 训练会删除全常量标签和单选 query，并默认禁用不可跨模型校准的 `main_*` type heads。

## 服务器运行

```bash
OUT=outputs/ogerpon_front100_$(date +%Y%m%d) \
WORKERS=48 MODEL_JOBS=12 TRAIN_PARALLEL=4 \
tools/run_ogerpon_front100_search.sh
```

Majkel 服务器入口 `tools/run_ogerpon_front100_majkel_server.sh` 当前默认使用物理 GPU 1 和 2，并显式调用 `/home/disk/HMZ/.venv/bin/python`。可用 `GPU_DEVICES=...` 覆盖；只有显式设置 `RESUME_INCOMPLETE=1` 才恢复未完成 artifact。

旧入口 `tools/run_ogerpon_single_teacher_server.sh` 默认转发到新流程；只有显式设置 `USE_LEGACY_SINGLE_TEACHER_PIPELINE=1` 才运行旧版同源审计流程。

正式运行前应先同步更多公开 keidroid episodes。当前仅60场数据时，结果可以筛选候选，但不能据此保证前100。

## 输出

- `manifests/`：可复现训练参数；
- `artifacts/`：artifact v2 与 episode split；
- `audits/`：锁定 test episode 行为一致率；
- `stage1/`、`stage2/`：legacy 快速淘汰；
- `holdout/`：严格 Kaggle profile 决赛；
- `submission_queue.json`：最多8个候选的线上验证顺序；
- `result.json`：固定 deck hash、排名和运行状态。

线上提交继续使用 `tools/kaggle_gold_loop.py` 的显式 `--execute` champion/challenger 流程。每个候选至少观察40场非自对战，建议100场后再晋升。
