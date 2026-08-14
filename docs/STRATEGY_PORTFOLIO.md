# 策略组合、牌组与历史表现

> **内部竞争性资料。** 本文包含教师来源、恢复牌表、候选路径、提交编号和私有评测结果，不适合直接发布到公开仓库。

本文是项目当前策略资产的固定快照。统计和排行榜数据截至 **2026-08-03**；Kaggle simulation 分数会随有效 episode 增加而变化，后续结果应以 Kaggle API 为准。

## 统计口径

为避免把模型参数或重复打包误算成新策略，本文使用以下口径：

| 项目 | 数量 | 统计方式 |
|---|---:|---|
| 核心策略家族 | 6 | 按独立战术体系统计 |
| 实际不同牌表 | 11 | 对自有提交包中的 `deck.csv` 内容去重 |
| 主要提交逻辑类型 | 4 | 原生规则、Meta Rules、Clone、Clone Search |
| 本地自有提交包 | 50 | 排除 `outputs/reference_submissions/` |
| 不同提交程序 | 36 | 对自有提交包中的 `main.py` 内容去重 |
| 本轮 Ogerpon 服务器候选 | 5 | 同一 keidroid 牌表的不同模型配置 |

服务器本轮五个 Ogerpon 包中，`balanced_v2` 已下载回本地，因此跨本地和服务器统计包数量时必须去重。外部下载的排行榜对手、公开 notebook submission 和锚点包不计入自有策略数量。

## 策略总览

| 家族 | 主要胜利路线 | 牌表数 | 当前判断 |
|---|---|---:|---|
| Great Tusk | 牌库破坏、资源干扰、Crustle 墙、应急 KO | 3 | 线上实绩最强，当前历史 champion |
| Mega Lucario | 快速进化、正面奖赏竞速、短程搜索 | 2 | 稳定进攻基准，近期缺少高分线上验证 |
| Grimmsnarl | 进化攻击、伤害计数器、后备目标控制 | 2 | 中等线上表现，不同评测池波动明显 |
| Mega Kangaskhan | 多属性能量、多攻击手、Ogerpon 引擎 | 1 | 当前最不成熟，不宜优先提交 |
| Mega Lopunny | Dudunsparce 循环、资源续航、攻击窗口管理 | 1 | 本地 holdout 潜力最高，尚未在线验证 |
| Teal Mask Ogerpon | Teal Dance 加速、多 Ogerpon 能量攻击 | 2 | 模仿一致率高，但 balanced 在线仅 600 |

## Great Tusk 牌库破坏

### 战术与实现

Great Tusk 是当前默认策略，也是规则、回放修复和服务器验证最完整的家族。默认牌表与战术入口分别位于 `deck.csv` 和 `main.py`。

核心路线包括：

- 使用 `Land Collapse` 破坏对手牌库；
- 使用 Explorer’s Guidance 强化牌库破坏回合；
- 使用 Dwebble / Crustle 建立防守墙；
- 使用 Xerosic’s Machinations、Boss’s Orders 等控制资源和场面；
- 根据双方牌库、奖赏和攻击准备情况切换牌库竞速、资源干扰、墙或 KO；
- 针对 Crustle、Mega Abomasnow 和自我牌库耗尽使用独立修复路线。

主要候选包括：

- `gt_anchor_robust_runtime`
- `gt_historical_disruption_deck`
- `gt_setup_first`
- `gt_anti_deckout`
- `gt_anti_crustle`
- `gt_anti_abomasnow`
- `gt_adaptive_race`
- `fixed_endgame_wall`

### 牌表

实际生成包中存在三套 Great Tusk 牌表：

1. 当前 Great Tusk / Dwebble / Crustle 标准牌表；
2. `submission_replay_restored_826.tar.gz` 使用的历史恢复牌表；
3. 早期 `submission.tar.gz` 使用的初始牌表。

### 优点

- 当前唯一经过大量服务器对战和多次 Kaggle 实测的家族；
- 不需要正面击倒所有高血量 Pokémon，拥有独特胜利条件；
- 规则透明，真实失败可以直接转化为条件、评分和回归测试；
- 对普通进攻牌组、资源不足牌组和部分慢速牌组具备稳定压力；
- `gt_anti_crustle` 拥有当前最可靠的自有线上历史成绩。

### 缺点

- 对双方牌库数量、剩余抽牌量和终局回合数非常敏感；
- 可选搜索牌和抽牌动作容易导致自己先牌库耗尽；
- 面对 Crustle 墙、Mega Abomasnow 和高压 KO 牌组需要专门分支；
- matchup-specific 规则持续增加后，规则之间可能互相干扰；
- 多个本地或服务器“改进”没有转化为更高 Kaggle 分数。

### 历史表现

| 候选 | 评测 | 结果 |
|---|---|---:|
| `gt_anti_crustle` | Kaggle submission `55128356` | **794.6** |
| `gt_anti_crustle` | 服务器 Stage 2 | 150-41-1 / 192 |
| `gt_anti_crustle` | 服务器 holdout | 150-89-1 / 240 |
| `gt_anti_crustle` | 600 局确认池 | 343-257 / 600 |
| `fixed_endgame_wall` | 600 局确认池 | 335-265 / 600 |
| `fixed_endgame_wall` | Kaggle submission `55132514` | 600.0 |
| `gt_legacy_826_anchor` | 历史标签 | 826.4 |
| `gt_legacy_826_anchor` | 重新提交 `55128354` | 678.6 |

`826.4` 是历史验证标签，不是 `55128354` 的实际重新提交分数。当前线上证据最强的 Great Tusk 版本仍是 `gt_anti_crustle`。

## Mega Lucario 进攻策略

### 战术与实现

Lucario 规则、隐藏世界采样和搜索位于 `agents/lucario_meta.py`。主要打法为：

- Riolu 进化 Mega Lucario ex；
- 使用 Fighting Gong、Premium Power Pro 和格斗能量快速形成攻击；
- 使用 Solrock、Hariyama 和 Trainer 牌维持场面；
- 根据对手使用 Battle Cage、Judge 等针对性组件；
- 对多个隐藏世界做短程 rollout，并使用风险惩罚聚合结果。

### 牌表

实际包中存在两套主要 Lucario 牌表：

1. `submission_lucario_guarded.tar.gz`；
2. `submission_lucario_replayfix_v1-v4.tar.gz` 使用的 replay-fix 牌表。

### 优点

- 胜利路线直接，以正面 KO 和奖赏竞速为主；
- 不需要处理 Great Tusk 式的复杂牌库竞速；
- 已具备隐藏信息采样、风险聚合和短程搜索；
- 适合作为 700 分级基准及通用进攻对手。

### 缺点

- 对起手、进化链和能量节奏依赖明显；
- Riolu 或关键能量被提前处理后恢复能力有限；
- 对墙、减伤、水系压力和资源干扰较敏感；
- 搜索开销高于纯规则策略；
- 没有近期且归属明确的高分线上提交。

### 历史表现

- `基准/submission_sorce_700.tar.gz` 被项目作为 700 分基准；
- Great Tusk 600 局确认池中，`submission_lucario_guarded` 为 237-243 / 480；
- 服务器 holdout 中，`submission_lucario_guarded` 为 77-82-1 / 160。

Lucario 当前更适合充当稳定进攻基准，而不是优先占用提交额度的 challenger。

## Grimmsnarl 回放克隆

### 战术与实现

Grimmsnarl 使用 Marnie’s Impidimp、Morgrem、Grimmsnarl ex、Froslass 和 Munkidori 构成进化攻击与伤害计数器体系。家族配置位于 `tools/build_meta_submission.py`，训练 artifact 包括 `outputs/meta_imitation/grim.json` 和 `outputs/meta_imitation/ntuml.json`。

当前实现已处理：

- 回放 observation/action 一步错位；
- Punk Up 必选路线；
- 缺少 Impidimp 时的立即恢复；
- 对手切换目标选择；
- Duraludon 进化前压制；
- Riolu 等关键进化线优先级。

### 牌表与提交

Grimmsnarl 有两套牌表：

1. `grim` 牌表；
2. `ntuml` 牌表。

两套牌表均可构建 `rules`、`clone` 和 `clone_search`。

### 优点

- 能同时处理正面攻击、伤害计数器和后备 Pokémon；
- 多教师数据覆盖的局面比单教师 Ogerpon 更广；
- 对关键进化线和脆弱后备目标具备明确压制路线；
- 某些 family screen 中表现明显优于 Great Tusk 候选。

### 缺点

- 多教师可能包含互相冲突的动作偏好；
- 对进化顺序、目标选择和场面恢复非常敏感；
- 不同评测池之间波动大；
- 私有 registry 对 submission `55188036` 存在重复映射。

### 历史表现

| 评测 | 结果 |
|---|---:|
| Final holdout | 27-21 / 48 |
| 一次 family screen | 35-13 / 48 |
| Broad holdout | 18-22 / 40 |
| 另一确认池 | 29-34-1 / 64 |
| Kaggle `55188036`，较早 registry 快照 | 608.3 |
| Kaggle `55188036`，2026-08-03 API 快照 | **668.6** |

`outputs/kaggle_gold/state.json` 同时把 `55188036` 关联到 `gt_anchor_robust_runtime` 和 `grim_clone`。根据提交时间与事件日志，该提交更可能是 Grim clone，但在修复 registry 前不应把该归属视为完全可靠。

## Mega Kangaskhan 多能量策略

### 战术与牌表

Kangaskhan 使用一套牌表，核心包含 Mega Kangaskhan ex、Teal Mask Ogerpon ex、Meowth ex、Raging Bolt ex、Latias ex、Crispin、Energy Switch 和 Area Zero Underdepths。

当前提交包括：

- `kang_rules`
- `kang_clone`
- `kang_clone_search`

### 优点

- 多攻击手和多属性能量提供理论上的对局适应性；
- Ogerpon 可以提供草能量加速和抽牌；
- 同时支持纯规则、行为克隆和短程搜索。

### 缺点

- 多属性能量管理和合法目标组合复杂；
- 错误 bench、附能或 Energy Switch 的代价很高；
- 当前规则层不足以稳定规划多回合资源；
- 当前是六个家族中实测最弱的家族之一。

### 历史表现

- Final family screen：6-41-1 / 48；
- Lopunny holdout 对手池：21-107 / 128；
- 本地 Kaggle 分估计约 319–479；
- 没有可靠的线上提交记录。

在动作表示和跨回合规划重构前，不建议为 Kangaskhan 使用 Kaggle 提交额度。

## Mega Lopunny 循环策略

### 战术与实现

Lopunny 使用 Mega Lopunny ex、Dunsparce / Dudunsparce、Fan Rotom 和 Trainer 引擎。核心难点不是单次动作，而是跨回合判断何时继续循环、何时攻击以及何时保护牌库。

该家族已加入 temporal memory，并提供四种 profile：

- `baseline`
- `attack`
- `cycle`
- `full`

这些 profile 属于同一牌表和策略家族的行为配置，不计为新牌组。

### 优点

- 当前本地 holdout 数据最亮眼的克隆家族；
- 循环建立后拥有较强的资源连续性；
- temporal memory 比单帧行为克隆更适合处理 Dudunsparce 循环；
- 多个模型参数版本在同一 holdout 中都取得高胜率。

### 缺点

- 过度循环会错过攻击窗口或导致牌库耗尽；
- `clone_search` 不一定优于纯 clone，错误价值函数可能破坏教师节奏；
- 不同评测池中的表现差异很大；
- 尚无真实 Kaggle 提交验证。

### 历史表现

Lopunny autoscreen holdout：

| 候选 | 结果 | 本地分数估计 |
|---|---:|---:|
| `lopunny_loss_aware_baseline` | 84-12 / 96 | 634.87 |
| `lopunny_balanced_baseline` | 87-9 / 96 | 610.92 |
| `lopunny_deep_baseline` | 85-11 / 96 | 602.31 |
| `lopunny_reference_cycle` | 86-10 / 96 | 595.43 |

其他评测池中：

- `lopunny_temporal_search`：17-47 / 64；
- `lopunny_timeline_v2`：17-47 / 64；
- `lopunny_clone`：25-39 / 64。

Lopunny 是最值得继续做严格服务器验证的未提交家族，但必须先解释和消除跨评测池波动。

## Teal Mask Ogerpon 单教师克隆

### 教师与战术

当前主要教师是 `keidroid`，回放目录为：

`outputs/kaggle_logs/top_leaders_20260802/keidroid_55153405`

该策略使用多只 Teal Mask Ogerpon ex，通过 Teal Dance 附草能量并抽牌，再使用 Myriad Leaf Shower 输出。另有 `Majkel1337` 和 combined 教师实验。

### 牌表

Ogerpon 有两套不同牌表：

1. keidroid 牌表：17 张草能量，并包含 Crushing Hammer、Tera Orb、Judge；
2. Majkel1337 牌表：18 张草能量，并提高 Energy Search / Retrieval 比例。

### 当前模型

| 模型 | 总语义一致率 | 特性一致率 | 审计结果 |
|---|---:|---:|---|
| `deep_v2` | **94.90%** | 97.54% | 通过 |
| `balanced_v2` | 93.73% | **97.81%** | 通过 |
| `winsoft_v2` | 93.18% | 97.26% | 通过 |
| `private` | 90.56% | 96.03% | 未通过 |
| `compact_v2` | 89.88% | 96.31% | 未通过 |

筛选门槛为总语义一致率至少 92.80%，特性一致率至少 93.71%。

### 优点

- 当前教师行为审计最完整、最可量化；
- 特性、附能和攻击行为一致率很高；
- 单教师数据避免了多教师动作冲突；
- `deep_v2` 已接近 95% 总语义一致率。

### 缺点

- 选卡、选能量和撤退仍存在关键错误；
- 训练准确率接近 100%，验证约 89%，过拟合明显；
- 无法恢复教师内部记忆、隐藏状态和未选择动作的反事实价值；
- 高行为一致率没有转化为高 Kaggle 分数。

### 历史表现

- `balanced_v2` 清洁包：`outputs/kaggle_manual_submit_20260803/ogerpon_balanced_v2_kaggle.tar.gz`；
- Kaggle submission：`55201683`；
- Kaggle 分数：600.0；
- 教师 `keidroid` 在 2026-08-03 排行榜快照中为第 6 名，1140.4；
- 三个通过行为门禁的候选正在服务器真实对战，本文快照时尚未产生 screen ranking。

下一次若继续该家族，应优先验证 `deep_v2`，而不是继续仅依据行为一致率提交 balanced 参数微调。

## 提交逻辑类型

### 原生规则

代表 Great Tusk 和 Lucario。提交直接运行手写评分、路线判断、安全规则及可选搜索，不依赖训练 artifact。

优点是可解释、容易从真实失败定位根因；缺点是规则数量增加后容易产生交互和回归。

### Meta Rules

通过 `agents/meta_runtime.py` 执行，但模型权重为零。代表 `grim_rules`、`ntuml_rules` 和 `kang_rules`，主要用于判断 learned model 是否真正带来收益。

### Clone

规则层与 LightGBM LambdaRank 行为克隆结合。代表 Grim、Kangaskhan、Lopunny 和 Ogerpon 的标准 clone 包，是当前回放模仿的主要提交类型。

### Clone Search

行为模型先给合法动作排序，再对少量候选进行短程 rollout。搜索可能纠正模型 Top-1 错误，也可能因为状态价值函数不准确而破坏教师行为。

Meta 提交的三个正式 `variant` 是 `rules`、`clone` 和 `clone_search`。Lopunny profile 是同一 variant 下的行为子配置；Kaggle-clean export 只移除内部元数据，不构成新策略类型。

## Kaggle 历史快照

截至 2026-08-03，能够明确读取的主要记录为：

| Submission | 策略 | 分数 | 说明 |
|---|---|---:|---|
| `55128356` | Great Tusk anti-crustle | **794.6** | 当前最强可靠自有历史记录 |
| `55128354` | Great Tusk legacy restore | 678.6 | 历史标签为 826.4，但重提结果较低 |
| `55188036` | 更可能是 Grim clone | **668.6** | registry 存在重复映射 |
| `55132514` | Great Tusk fixed endgame wall | 600.0 | 本地通过门禁但线上退化 |
| `55201683` | Ogerpon balanced_v2 | 600.0 | 高模仿一致率未转化为榜分 |

2026-08-03 11:56（北京时间）的公开榜快照：

- 队伍：`DarkLayer`；
- 当前队伍分数：668.6；
- 当前排名：2699；
- keidroid：第 6 名，1140.4。

Simulation competition 的 submission 分数会随有效 episode 更新。表中数据是历史快照，不应被当作永久最终分数。

## 综合判断

按当前证据排序：

1. **Great Tusk anti-crustle**：线上实绩最强，仍是实际 champion；
2. **Lopunny loss-aware / balanced**：本地 holdout 潜力最高，但没有在线验证；
3. **Grim clone**：中等线上表现，跨评测池波动明显；
4. **Ogerpon deep_v2**：行为复刻质量最高，但真实强度仍待服务器验证；
5. **Lucario guarded / replayfix**：稳定基准，但近期没有突破性成绩；
6. **Kangaskhan clone**：当前明显不成熟。

最重要的历史结论是：**本地胜率、行为一致率和 Kaggle 分数之间的相关性仍然较弱。** Great Tusk 的多个本地改进在线从 794.6 退化到 600，Ogerpon 93.73% 的教师行为一致率也只得到 600。因此后续候选必须优先通过广泛服务器真实对战、独立 holdout 和线上小规模验证，不能仅凭 smoke test 或模仿准确率晋升。

## 数据来源与维护

本文快照主要依据：

- `outputs/kaggle_gold/state.json`
- `outputs/great_tusk_gold/**/ranking.json`
- `outputs/great_tusk_fixed_confirmation/holdout/ranking.json`
- `outputs/meta_imitation/**/summary.csv`
- `outputs/teacher_clone_audit/*.json`
- 服务器 `outputs/ogerpon_single_teacher_20260803/audit_summary.json`
- Kaggle API leaderboard 与 team submission 查询

更新本文时必须：

1. 对 `deck.csv`、`main.py` 和提交包按内容哈希去重；
2. 区分 Kaggle 真实分数、本地 TrueSkill 估计和行为一致率；
3. 为动态排行榜数据记录绝对日期；
4. 检查 submission ID 是否在 registry 中重复映射；
5. 不写入 Kaggle token、API key 或原始 replay 内容。
