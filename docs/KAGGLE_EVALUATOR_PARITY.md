# Kaggle Evaluator Parity

本地评估器现在提供两个显式 profile：

- `legacy`：保留原有快速超时、较小步数上限和可选公共调度种子，适合 smoke test 与早期搜索；
- `kaggle`：冻结 Kaggle `kaggle_environments==1.16.0` 的公开 CABT 参数与状态语义，适合最终 holdout 和晋升门禁。

## 严格 Profile

CLI 默认使用 `kaggle`。严格 profile 固定为：

```text
actTimeout=0
remainingOverageTime=600
runTimeout=2000
episodeSteps=10000000
```

`cg` 的关键 Python 文件与 Linux 原生库会在报告中记录 SHA-256，并与冻结的官方样例哈希比较。报告同时声明 `engine_seed_control=false`：公开 `BattleStart` 使用原生 `std::random_device`，Python seed 无法控制洗牌和隐藏状态。因此严格 profile 会拒绝 `common_random_seeds=True`。

对战报告新增双方最终 `status`、`reward`、`failure_class` 和 `ranking_eligible`。Agent 的 `ERROR`、`INVALID`、`TIMEOUT` 按官方 core 在 interpreter 后把故障方 reward 归零为 `null`；外层 watchdog 或裁判进程故障标记为 `HARNESS_FAILURE`，不进入 TrueSkill 排名。

## 门禁分层

- `tools/server_gold_autoscreen.py`：stage1/stage2 使用 `legacy`，最终 holdout 使用 `kaggle`；
- `tools/robust_gold_search.py`：搜索和 holdout 使用 `legacy`，final 使用 `kaggle`；
- `tools/server_candidate_eval.py` 与 `tools/gold_gate.py`：显式使用 `kaggle`。

## 排名校准

私有配置使用 `configs/kaggle_eval_calibration_private.json`，该路径已加入 `.gitignore`。字段规范见 `configs/kaggle_eval_calibration.schema.json`。不要写入 Kaggle token、原始 replay payload 或未确认的提交映射。

```bash
python -m tools.kaggle_eval_calibration validate
python -m tools.kaggle_eval_calibration sync --kaggle kaggle
python -m tools.kaggle_eval_calibration evaluate --workers 48
python -m tools.kaggle_eval_calibration calibrate
```

`sync` 会从真实 episode 自动派生对手频率、先后手频率和未知对手覆盖率；也可以在私有配置中提供人工 opponent/seat 权重作为回退。校准以候选排序为主：Spearman 至少 `0.80` 且 pairwise order accuracy 至少 `80%` 才允许自动提交。少于 5 个已确认且 public score 不同的历史提交时，只输出 `kaggle_rank_score`，不生成可用的 public score 预测。已知重复 submission ID `55188036` 必须放入 `excluded_submission_ids`，直至人工确认其包哈希映射。

## 无法本地复刻的部分

公开运行语义、引擎二进制和超时预算可以冻结；Kaggle 隐藏的匹配池、对手抽样、并发环境及排行榜 rating 参数不能完全复制。因此最终目标是让单局裁判行为一致，并通过历史真实提交校准候选排序，而不是声称本地分数等于 Kaggle public score。
