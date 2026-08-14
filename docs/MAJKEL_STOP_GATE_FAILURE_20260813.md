# Majkel v8 停止门控失败深度分析（2026-08-13）

## 结论摘要

Majkel v8 已经排除了“没有使用新数据”“旧数据重复计权”“GPU 没有训练”“停止头容量太小”等表层问题，但仍未通过严格停止门控。

本轮使用：

- 最新提交 283 局；
- 旧同牌组数据中独有的 80 局；
- 按回放内容哈希去重后共 363 局；
- 19,524 条行为决策；
- 物理 GPU0；
- CatBoost full-feature 决策级停止分类器；
- Continue 样本权重 1.35；
- 3 个不同 seed 的模型平均集成；
- 胜局权重 1.0、败局权重 0.35；
- 按 episode 分组的稳定 train/validation/test 切分。

最终验证集最优结果：

| 指标 | 实际结果 | 严格要求 | 结论 |
| --- | ---: | ---: | --- |
| Continue recall | 85.15% | ≥90% | 失败 |
| Terminal recall | 84.73% | ≥70% | 通过 |
| Balanced accuracy | 84.94% | ≥85% | 失败 |

失败的核心不是阈值选择，而是两类分数在一个特定状态子空间内发生实质重叠：**场上已经存在攻击，尤其已经存在击倒攻击，但 Majkel 仍选择继续出牌、发动能力、撤退、贴能或进化。**

在没有攻击可用的状态中，模型 Continue recall 达到 95.71%；一旦攻击可用，Continue recall 降到 79.13%；存在击倒攻击时进一步降到 68.13%。因此，当前停止模型实际学到的近似规则仍然是“攻击越成熟，越应该终止”，而 Majkel 的真正策略包含大量“攻击已经足够好，但继续优化场面仍然更好”的行为。

这也是为什么继续增加普通数据、全局提高 Continue 权重或继续扫描阈值不能彻底解决问题。

## 数据审计

### 新旧回放去重

最新目录包含 283 局，旧同牌组冻结目录包含 319 局。按解压后的规范化 JSON 内容计算 SHA-256：

- 两个目录重复 239 局；
- 旧目录独有 80 局；
- 最新目录独有 44 局；
- 合并后共 363 个唯一回放。

训练目录：

```text
outputs/opponent_league_20260804/replays/majkel_unique363_20260813
```

训练清单：

```text
outputs/opponent_league_20260804/manifests/majkel_unique363_gpu0_v8.json
```

没有把重叠的 239 局重复训练，因此本轮新增数据不会通过重复计权伪造样本量。

### 解析和切分

363 局全部成功解析：

- resolved：363；
- ambiguous：0；
- deck mismatch：0；
- 总决策：19,524。

按 episode 分组切分：

| Split | Episodes | Decisions |
| --- | ---: | ---: |
| Train | 255 | 13,690 |
| Validation | 54 | 2,803 |
| Test | 54 | 3,031 |

停止门控只评估同时存在继续和终止候选的 MAIN 决策。验证集包含：

- Continue：1,414；
- Terminal：262；
- 合计：1,676。

这说明失败不是由少量十几个样本造成。验证集中有 262 个真实终止样本和 1,414 个真实继续样本，足以稳定暴露当前边界问题。

## 本轮模型结构

停止头已经从旧的“类型排序器”改成真正的决策级二分类器。每个停止样本同时输入：

1. 最强 Continue 类型的完整聚合特征；
2. 最强 Terminal 类型的完整聚合特征；
3. Terminal 与 Continue 的逐特征差值。

训练配置：

```text
CatBoostClassifier
loss_function = Logloss
eval_metric = AUC
router_n_estimators = 650
router_num_leaves = 63
router_min_child_samples = 32
router_l2_leaf_reg = 10.0
stop_continue_weight = 1.35
stop_ensemble_size = 3
```

三个模型 seed：

```text
20260813
20261822
20262831
```

三个模型均在物理 GPU0 完成训练，最后通过平均 raw score 组成集成停止头。

因此，本轮失败不能解释为“停止头仍在使用旧的错误 ranker”或“只加载了旧模型”。

## 阈值不可达证明

### 最佳 Balanced Accuracy 阈值

验证集全部候选阈值中，Balanced Accuracy 最高点为：

```text
threshold = -1.8325772044
continue recall = 0.8514851485
terminal recall = 0.8473282443
balanced accuracy = 0.8494066964
```

混淆矩阵：

| 实际类别 | 正确 | 错误 |
| --- | ---: | ---: |
| Continue | 1,204 | 210 次 false-stop |
| Terminal | 222 | 40 次 false-continue |

### 强制满足 Continue ≥90% 的反事实

如果将阈值提高到能够满足 Continue recall ≥90% 的最佳位置：

```text
threshold = -1.3708580531
continue recall = 0.9009900990
terminal recall = 0.7748091603
balanced accuracy = 0.8378996297
```

此时：

- Continue 门控通过；
- Terminal 门控通过；
- Balanced accuracy 仍只有 83.79%，无法达到 85%。

这证明不能通过“再调一点阈值”解决问题。为了在 Continue 为 90.10% 时达到 85% Balanced Accuracy，Terminal recall 至少需要约 79.90%，但当前该阈值只有 77.48%。

### 分数分布重叠

Continue margin 分位数：

| 分位数 | Margin |
| --- | ---: |
| 10% | -5.372 |
| 25% | -4.714 |
| 50% | -3.769 |
| 75% | -2.545 |
| 90% | -1.384 |
| 最大值 | 2.470 |

Terminal margin 分位数：

| 分位数 | Margin |
| --- | ---: |
| 10% | -2.409 |
| 25% | -1.298 |
| 50% | -0.197 |
| 75% | 1.132 |
| 90% | 1.992 |
| 最大值 | 3.320 |

Terminal 的低分尾部和 Continue 的高分尾部明显交叠。单一全局阈值必须在 false-stop 与 false-continue 之间交换，无法同时满足三个严格门控。

## False-stop 具体发生在哪里

在最佳 Balanced Accuracy 阈值下，共有 210 次 false-stop，即 Majkel 实际要继续，但模型认为应攻击或结束。

### 按教师真实动作

| Majkel 实际动作类型 | False-stop 数 | 占全部 False-stop |
| --- | ---: | ---: |
| PLAY | 154 | 73.33% |
| ABILITY | 24 | 11.43% |
| RETREAT | 13 | 6.19% |
| ATTACH | 12 | 5.71% |
| EVOLVE | 7 | 3.33% |

主要故障不是模型错过少见动作，而是**大量把 Majkel 的继续出牌错误截断**。

### 攻击是否可用

| 状态 | Continue Recall | Terminal Recall | Balanced Accuracy |
| --- | ---: | ---: | ---: |
| 没有 ATTACK 候选 | 95.71% | 83.72% | 89.72% |
| 存在 ATTACK 候选 | 79.13% | 84.93% | 82.03% |

210 次 false-stop 中：

- 188 次存在攻击候选，占 89.52%；
- 只有 22 次不存在攻击候选。

模型在“没有攻击可选”时已经通过全部严格门控。真正失败区域几乎完全是“已经可以攻击，但 Majkel 还不想停”。

### 是否存在击倒攻击

| 状态 | Continue Recall | Terminal Recall | Balanced Accuracy |
| --- | ---: | ---: | ---: |
| 没有可击倒攻击 | 90.13% | 80.42% | 85.27% |
| 存在可击倒攻击 | 68.13% | 89.92% | 79.02% |

存在击倒攻击时：

- Continue 样本 320 个；
- 正确继续 218 次；
- false-stop 102 次；
- 每约 3.14 个应继续的状态就提前终止 1 次。

这组数据单独解释了近一半 false-stop。Majkel 的策略并不是“能击倒就立刻攻击”，而是会根据奖赏、后续回合资源、场面完整度、剩余可执行动作和对手状态决定是否先继续优化。

### 回合阶段

| 回合 | Continue Recall | Terminal Recall | Balanced Accuracy | False-stop |
| --- | ---: | ---: | ---: | ---: |
| 1–2 | 93.33% | 95.35% | 94.34% | 24 |
| 3–4 | 89.66% | 75.00% | 82.33% | 36 |
| 5–7 | 85.39% | 82.61% | 84.00% | 51 |
| 8+ | 72.27% | 86.27% | 79.27% | 99 |

模型在前两回合表现很好，晚期持续退化。第 8 回合及以后贡献 99/210，即 47.14% 的 false-stop。

晚期状态中攻击普遍成熟、场面信息复杂、资源和奖赏价值高度依赖具体局势，单纯把“攻击强度”解释为终止意图会系统性偏向提前攻击。

### 本回合已经执行的动作数

210 次 false-stop 中：

| 本回合已执行动作数 | False-stop |
| --- | ---: |
| 0–1 | 10 |
| 2–3 | 24 |
| 4–6 | 43 |
| 7+ | 133 |

63.33% 的 false-stop 发生在本回合已经执行至少 7 个动作之后。

这不是简单的“模型不会开局展开”，而是模型无法判断长连招在什么时候真正完成。Majkel 经常在执行很多动作、攻击也已经可用之后继续榨取额外价值；模型则把“动作已经很多 + 攻击已经成熟”误认为可靠的终止信号。

### 胜负轨迹

210 次 false-stop 中有 123 次来自最终胜局，占 58.57%。

因此 false-stop 不能主要归因于败局中的噪声或绝望操作。大量被模型截断的继续动作来自 Majkel 最终获胜的轨迹，正是需要学习的高价值策略。

## 测试集复现

将验证集选择的阈值原样应用于完全锁定的 test episodes：

```text
continue recall = 0.8316961362
terminal recall = 0.8625429553
balanced accuracy = 0.8471195458
```

混淆矩阵：

| 实际类别 | 正确 | 错误 |
| --- | ---: | ---: |
| Continue | 1,270 | 257 次 false-stop |
| Terminal | 251 | 40 次 false-continue |

验证集和测试集结果方向一致：

- Continue recall 都在 83%–85%；
- Terminal recall 都在 85%–86%；
- Balanced accuracy 都略低于 85%。

因此不是某个验证切分偶然不利，也不是阈值对验证集过拟合。

## 模型实际依赖了什么

三个 stop classifier 合并后的高频树分裂特征包括：

1. Continue 候选的 `base_score_max`；
2. Terminal 候选的 `base_score_max`；
3. Terminal/Continue 的候选数量；
4. base score 的均值、标准差、第二和第三高分；
5. Terminal 与 Continue 的 base score 差值；
6. 奖赏数量；
7. 本回合动作数量；
8. 最近动作类型；
9. 本回合出现过的卡牌哈希；
10. 对手手牌和牌库数量。

模型确实使用了时序特征，并非完全看不到本回合历史。但最高频的决策依据仍然是已有 option/base 模型产生的分数及候选数量。

这会产生一个反馈问题：

1. 基础模型认为攻击是一个高价值合法动作；
2. stop classifier 大量使用基础分数判断终止；
3. 攻击越强，stop classifier 越倾向 terminal；
4. 但教师的真实决策可能是先 PLAY/ABILITY/RETREAT 再攻击；
5. stop classifier 因此重复放大基础模型对即时攻击价值的偏好，而没有独立学会“连招是否完成”。

## 为什么新增 80 局没有解决

从 283 局增加到 363 局后，停止模型的最佳 Balanced Accuracy 从 84.85% 变为 84.94%，只提高约 0.09 个百分点。Continue recall 从 83.17% 提高到 85.15%，但仍距离 90% 很远。

原因包括：

1. 新增 80 局来自同一历史提交，提供了更多样本，但没有专门增加“可击倒但仍继续”的边界样本密度；
2. 普通 Continue 样本已经较容易，继续增加它们对关键子空间帮助有限；
3. 最困难状态依赖具体牌序、剩余资源、奖赏路线和长连招意图，普通同分布数据不能快速分离边界；
4. 全局 Continue 权重会同时推动所有 Continue 样本，而真正需要额外权重的是 attack-ready/KO-ready 的 hard-continue 子集；
5. 阈值校准会抵消一部分全局类别权重变化，因此全局权重无法替代局部判别能力。

## 为什么三模型集成仍失败

三 seed 集成可以降低 CatBoost 随机波动，但不能修复共同的系统性偏差。三个模型使用：

- 相同数据；
- 相同标签；
- 相同特征；
- 相同损失；
- 相同 base score 输入。

当所有模型都把“成熟攻击”当成终止强信号时，平均分数只会得到更稳定的错误边界，而不会自动学会 Majkel 的长期意图。

集成解决的是方差问题；当前主要是偏差和任务分解问题。

## CatBoost 是否已经到极限

目前不能得出“必须换深度模型”的结论。

证据是：

- 无攻击候选区域已经达到 89.72% Balanced Accuracy；
- 无击倒攻击区域达到 85.27% Balanced Accuracy，并同时达到 Continue ≥90%；
- 前两回合达到 94.34% Balanced Accuracy；
- full-feature classifier 明显改变了 Terminal/Continue 权衡；
- test 与 validation 稳定一致。

这说明 CatBoost 能学习大部分停止决策。失败集中在明确、可定义的子任务：**attack-ready，尤其 KO-ready 状态中的继续意图。**

在该子任务被单独建模前，直接换 Transformer 或 LSTM 很可能只是在 363 局上引入更高方差和更强记忆能力，不能保证解决标签边界和任务分解错误。

## 根因排序

### 1. 任务分解错误

当前模型直接做一个全局 Continue/Terminal 二分类，但“无攻击时是否继续”和“攻击已经成熟时是否继续”是难度完全不同的任务。

前者近似规则问题，后者是长期价值和连招完成度问题。把两者交给同一阈值和同一分类器，会让容易样本主导损失，困难子空间被平均掉。

### 2. Hard-continue 样本没有被定向训练

当前 `stop_continue_weight=1.35` 对全部 Continue 样本生效。真正需要提高权重的是：

- ATTACK 可用但教师选择 Continue；
- KO 攻击可用但教师选择 Continue；
- 回合动作数较高但教师继续；
- 晚期回合仍继续出牌或发动能力；
- 最终胜局中的上述状态。

全局 Continue 加权会浪费大量权重在已经能正确分类的简单样本上。

### 3. 基础动作分数泄漏了即时攻击偏好

stop classifier 最常用的特征是 Continue/Terminal 的 `base_score_max`。基础动作模型的目标是选择教师动作，不是估计“连招是否结束”。把基础分数作为停止头最强输入，容易让停止头把“攻击动作看起来很强”误认为“现在应该立即攻击”。

### 4. 缺少直接表示未完成计划的特征

虽然已有 turn history，但仍缺少更直接的语义特征，例如：

- 当前仍有多少高置信 Continue 动作；
- 是否存在未使用且有正收益的能力；
- 是否仍有可执行搜索、抽牌、回收、换位或资源转换；
- 本回合场面价值是否仍在上升；
- 当前攻击与继续一步后的预期攻击差；
- 当前击倒是否会错过更高奖赏路线；
- 教师常见连招模板完成度。

CatBoost 无法从有限回放中稳定自行组合出这些高层语义。

### 5. 单一全局阈值不适合异质子空间

无攻击、普通攻击、击倒攻击三个子空间的最优边界明显不同。单一阈值必须折中：

- 对无攻击状态，当前模型已经足够保守；
- 对 KO-ready 状态，当前模型严重过早终止；
- 简单移动全局阈值会增加 false-continue，导致 Balanced Accuracy 下降。

## 下一版修复建议

### 第一优先级：两阶段 CatBoost 路由

保持 CatBoost，不降低门控，将停止问题拆成：

1. `attack_available == false`：使用简单 stop 模型或直接保持 Continue 强竞争；
2. `attack_available == true && ko_available == false`：普通 attack-ready 专家；
3. `ko_available == true`：KO-ready hard-continue 专家。

三个专家分别校准，但最终仍输出有限软 gate，任何合法动作不得设为 `-inf`。

### 第二优先级：定向 hard-negative mining

先用当前 v8 模型对训练集打分，只提高以下 false-stop 或高 margin Continue 样本的权重：

```text
actual = Continue
attack_available = true
且 stop_margin 接近或超过当前阈值
```

建议分层权重：

- 普通 Continue：1.0；
- attack-ready Continue：1.5；
- KO-ready Continue：2.0；
- 当前模型实际 false-stop：再乘 1.25–1.5；
- 胜败权重仍保持 1.0 / 0.35。

权重必须写入 artifact 审计，禁止再次全局阶段平衡。

### 第三优先级：减少 base score 支配

训练独立 stop 专家时做消融：

1. 完整特征；
2. 去除 `base_score_*`；
3. 只保留 `delta::base_score_*`；
4. 限制 base score 特征权重或通过 feature subsampling 降低依赖。

选择标准不是 AUC，而是 episode-grouped validation 上是否存在同时满足三项严格门控的阈值。

### 第四优先级：加入连招完成度特征

优先加入可确定计算、无需深度模型的特征：

- 剩余合法 Continue 类型数；
- 剩余 PLAY/ABILITY/ATTACH/EVOLVE/RETREAT 候选数；
- 本回合不同 Continue 类型已执行次数；
- 当前最强 Continue 与最强 Terminal 的 option margin；
- 最近一次 Continue 后攻击伤害是否提高；
- 本回合攻击伤害增长量；
- 当前是否存在未使用能力；
- 当前是否存在可增加场上总能量的动作；
- 当前是否存在可改变奖赏路线或攻击目标的动作。

### 第五优先级：只有 CatBoost 专家仍失败时再上小型深度模型

如果完成以下实验后，episode-grouped 多折验证仍无法通过门控：

- attack-ready/KO-ready 专家；
- hard-continue mining；
- base score 消融；
- 连招完成度特征；
- 多 seed CatBoost 集成；

再考虑小型序列 residual head。深度模型应只预测 stop residual，不应替换已经表现较好的 option/type heads。

## 下一轮验收标准

不得降低现有阈值：

```text
Continue recall >= 0.90
Terminal recall >= 0.70
Balanced accuracy >= 0.85
```

此外增加子组诊断，不作为替代门控：

```text
attack_available Continue recall >= 0.87
KO-ready Continue recall >= 0.80
turn 8+ Continue recall >= 0.82
```

只有全局严格门控通过后，才允许：

1. 生成 artifact；
2. 构建候选包；
3. 进行 32/64/256/512 局闭环晋升；
4. 加入对手池。

本轮仍不得上传或提交 Kaggle。

## 可复现产物

诊断工具：

```text
tools/majkel_stop_failure_analysis.py
```

远端完整 JSON 报告：

```text
outputs/opponent_league_20260804/audits/majkel_unique363_gpu0_v8.stop_failure.json
```

训练日志：

```text
outputs/opponent_league_20260804/majkel_unique363_gpu0_v8.log
```

停止模型部件：

```text
outputs/opponent_league_20260804/artifacts/majkel_unique363_gpu0_v8_parts/
```

由于严格门控失败，没有生成：

```text
outputs/opponent_league_20260804/artifacts/majkel_unique363_gpu0_v8.json
```

也没有生成或提交 Kaggle 候选包。
