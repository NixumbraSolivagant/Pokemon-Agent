# Kaggle Replay 驱动的策略优化

本轮优化不再把本地 ladder 胜率当作 Kaggle 榜分的直接代理，而是从真实提交对局中定位可复现的决策错误，再把这些错误转化为策略规则、牌组变体和回归测试。

## 数据与边界

- 分析了 116 场已下载的 Kaggle episode replay，覆盖多个提交版本和主要对手类型。
- 排除自对局后，重点复盘了一个代表性提交的 6 场失败：5 场由奖赏竞速或击倒失败导致，1 场由牌库耗尽导致。
- 原始 replay、下载报告、恢复出的对手牌表和排行榜牌表不进入 Git；它们可能包含大量实验数据或竞争性信息。
- `tools/kaggle_replay_miner.py` 只负责离线读取已经下载的 episode 数据，不保存 Kaggle token，也不在本机运行模拟对局。

## 发现的根因

### Hariyama 进化被错误抑制

旧策略在多个真实状态中拥有可进化的 Makuhita，却直接选择攻击。原因是进化评分把 Hariyama 过度绑定到 Crustle 对局，同时主阶段候选扫描还错误地把该进化标记为具备 gust 能力。

修复后：

- 允许非 Crustle 对局按场面价值进化 Hariyama；
- 删除错误的 gust 标记；
- 提高已投入能量的 Makuhita/Hariyama 进化线价值；
- 在提取出的 6 个失败状态上，策略从 6 次直接攻击改为 6 次优先进化。

### Great Tusk 对局发生自我牌库耗尽

真实败局中，策略在牌库已经很薄时仍连续使用 Dusk Ball、Fighting Gong 等压缩牌库的卡，最终先于对手耗尽。

修复后：

- 识别 Great Tusk/Crustle 类牌库破坏对局；
- 使用更保守的 `MILL_DECK_RESERVE` 阈值；
- 低牌库时抑制 Dusk Ball、Fighting Gong、Poké Pad、Carmine 和 Lillie’s Determination；
- 保留可以立即改变胜负条件的攻击、进化和资源动作。

### Alakazam 手牌伤害缺少干扰

真实对局中，对手手牌持续增长并直接转化为攻击伤害。原牌组没有可靠的手牌重置手段。

修复后：

- `configs/lucario_replayfix_deck.csv` 用 1 张 Judge 替换 1 张 Dusk Ball，牌组仍为合法 60 张；
- Judge 被加入关键搜索候选，面对大手牌或已识别的 Alakazam 体系时提高优先级；
- 减少一张 Dusk Ball 也降低了牌库破坏镜像中的自我压缩风险。

### 对手威胁估值过于粗糙

旧策略对未专门建模的攻击手主要使用“能量数量 × 固定权重”的默认估值，无法区分准备完成的关键攻击手和普通铺场 Pokémon。

修复后新增 Archaludon、Duraludon、Cinderace、Hydrapple 和 Great Tusk 的攻击压力估值，并将核心进化线缺失作为对手模型匹配的重罚项。

## 新增工作流

### Replay 分析器

```bash
python3 -m tools.kaggle_replay_miner \
  --log-root output/kaggle_logs \
  --submission-ids SUBMISSION_ID \
  --out output/kaggle_logs/report
```

输出包括：

- `episodes.csv`：逐场结果、先后手、终局资源、对手类型和失败终结动作；
- `summary.json`：按提交、对手类型和先后手聚合的胜负统计；
- 可继续用于提取失败状态并进行反事实动作验证。

### Lucario 搜索与牌组变体

`tools/lucario_meta_search.py` 生成多种合法牌组和搜索配置，覆盖：

- Judge、Battle Cage、Legacy Energy 等针对性组件；
- Hariyama 数量、能量数量、Switch 和 Boss’s Orders 数量变化；
- heuristic、fast、balanced、guarded、deep 五类搜索预算；
- 多对手模型隐藏世界采样与风险惩罚。

`tools/build_meta_proxies.py` 提供可公开构造的 archetype proxy；`tools/robust_gold_search.py` 使用最差对局、Wilson 下界和 CVaR 等稳健指标，避免只优化平均胜率。

## 验证原则

本地评测只用于发现明显退化和动作错误，不用于承诺 Kaggle 分数。候选晋升遵循以下顺序：

1. 在真实失败状态上验证动作是否修正；
2. 检查 60 张牌组合法性和 Kaggle `exec` 导入兼容性；
3. 在服务器上运行针对性单元测试和必要的模拟评测；
4. 通过真实 Kaggle 提交验证分数；
5. 将新 replay 继续加入失败分析，避免围绕一次公开分数陷入局部最优。

## 本轮文件变更

- `agents/lucario_meta.py`：Lucario 规则策略、搜索、隐藏世界采样和 replay 根因修复。
- `configs/lucario_replayfix_deck.csv`：加入 Judge 的 60 张修正版牌组。
- `tools/kaggle_replay_miner.py`：真实 Kaggle replay 离线聚合工具。
- `tools/lucario_meta_search.py`：Lucario 牌组与搜索配置候选生成、评测和导出。
- `tools/build_meta_proxies.py`：主要 archetype 的合法公开 proxy 牌组。
- `tools/robust_gold_search.py`：面向最差对局表现的稳健候选筛选。
- `tests/test_lucario_meta.py`、`tests/test_meta_proxies.py`：运行时渲染、无 `__file__` 导入、牌组合法性和 proxy 回归测试。
- `main.py`、`tests/test_main.py`：修复 Kaggle `exec` 环境下的牌组路径读取，并补充 Dragapult 防守组件逻辑。

这些改动提高的是策略对真实失败模式的覆盖度，并不构成金牌或固定榜分保证。后续优化应继续以新增真实 replay 和隐藏测试表现为准。
