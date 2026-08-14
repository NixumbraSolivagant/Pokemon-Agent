# Majkel v9 停止路由实施说明

## 目标

v9 仅在 `Majkel1337` 训练中启用 schema v2。它修复 v6 的重复平衡/硬路由问题，并针对 v8 的 attack-ready、KO-ready false-stop 边界增加定向训练与门禁。

## 训练机制

- `main_stop_no_attack`、`main_stop_attack_ready`、`main_stop_ko_ready` 各自训练三 seed CatBoost 集成。
- Continue 权重为 no-attack `1.0`、attack-ready `1.5`、KO-ready `2.0`；胜负权重 `1.0/0.35`，最终样本权重封顶 `2.70`。
- 仅使用 train episodes 做 5-fold OOF false-stop mining；被命中的 Continue 样本额外乘 `1.35`。
- 每个专家比较 full/no-base/delta-only/damped-base 四种停止特征消融；所选模式写入 artifact 并由 runtime 同步应用。
- runtime 使用有限软偏置、置信带和 ensemble 方差；任何合法动作不得被赋值 `-inf`。
- 反事实 `residual_q` 只在停止低置信时生效，贡献裁剪为 `[-0.5, 0.5]`。

## 冻结与运行

先在服务器创建冻结清单一次：

```bash
python -m tools.freeze_replay_dataset \
  --replays outputs/opponent_league_20260804/replays/majkel_unique363_20260813 \
  --target-name Majkel1337 \
  --out outputs/opponent_league_20260804/manifests/majkel_unique363_v1.frozen.json \
  --seed 20260813
```

启动训练必须显式提供两个路径；launcher 会先校验回放内容 SHA-256 和 episode split：

```bash
REPLAYS=outputs/opponent_league_20260804/replays/majkel_unique363_20260813 \
FROZEN_DATASET=outputs/opponent_league_20260804/manifests/majkel_unique363_v1.frozen.json \
SPLIT_SEED=20260813 \
bash tools/run_ogerpon_front100_majkel_server.sh
```

## 晋升门禁

validation 与冻结 test 均要求 Continue ≥92%、Terminal ≥75%、Balanced Accuracy ≥87%、attack-ready Continue ≥89%、KO-ready Continue ≥85%、turn 8+ Continue ≥85%。任一失败不生成完整 artifact 或候选包。
