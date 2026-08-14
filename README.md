# Pokémon TCG AI Agent

私有 champion/challenger 流程见 `docs/KAGGLE_GOLD_LOOP.md`；keidroid Ogerpon 前100搜索见 `docs/OGERPON_FRONT100_SEARCH.md`，教师初始化残差 Q、真实 CG belief search 与 CEM 闭环见 `docs/OGERPON_RESIDUAL_OPTIMIZATION.md`；排行榜回放大池见 `docs/OPPONENT_LEAGUE.md`；评估器一致性与排行榜校准见 `docs/KAGGLE_EVALUATOR_PARITY.md`；完整策略、牌组、提交类型和历史表现见 `docs/STRATEGY_PORTFOLIO.md`。竞争牌组、回放模型和提交状态默认保存在已忽略路径中。

一个面向 **Pokémon TCG AI Battle / Kaggle 风格对战环境** 的完整策略研发项目。

本仓库不只是一个可提交的 `agent()`，还包含：

- Great Tusk 牌库破坏与 Crustle 防守体系的规则策略；
- 基于官方原生模拟器搜索接口的短程决策增强；
- 隐藏信息信念采样与风险惩罚；
- 隔离进程、本地超时裁决和 TrueSkill 排名评测；
- 候选牌组、策略参数和对手模型的自动生成；
- Sequential Racing、PSRO、失败样本挖掘和多阶段 holdout 筛选；
- Kaggle 提交压缩包的构建、清理和导出工具。

> 当前主策略以 Great Tusk 的牌库破坏能力为核心，搭配 Crustle 防守墙、资源干扰及应急攻击手。项目结构允许自动生成其他完整合法牌组家族并进行对手池内竞争。

## 目录

- [项目特点](#项目特点)
- [Replay 驱动优化](#replay-驱动优化)
- [策略组合与历史表现](#策略组合与历史表现)
- [策略概览](#策略概览)
- [架构](#架构)
- [运行要求](#运行要求)
- [快速开始](#快速开始)
- [构建提交包](#构建提交包)
- [本地评测](#本地评测)
- [自动策略发现](#自动策略发现)
- [其他优化工作流](#其他优化工作流)
- [牌组与提交格式](#牌组与提交格式)
- [测试与验证](#测试与验证)
- [输出文件](#输出文件)
- [远程运行脚本](#远程运行脚本)
- [开发约定](#开发约定)
- [常见问题](#常见问题)
- [仓库状态说明](#仓库状态说明)

## 项目特点

### 比赛 Agent

- 所有公开决策最终由 `main.py` 中的 Kaggle 兼容 `agent(obs_dict, configuration=None)` 返回。
- 针对出牌、进化、贴能、撤退、攻击和卡牌目标选择分别计算动作分数。
- 根据双方场面、资源和可见牌切换进攻、防守墙及牌库竞速模式。
- 所有公开 Agent 异常都回退到统一的合法动作选择逻辑，降低线上崩溃风险。
- `deck.csv` 相对于脚本目录读取，不依赖调用方当前工作目录。

### 搜索增强

- 使用 `cg.search` 对高价值候选动作进行短程 rollout。
- 基础规则策略作为搜索先验，避免在所有合法动作上平均浪费预算。
- 对隐藏手牌和牌库进行可复现采样。
- 使用多世界聚合和风险惩罚，避免只对单个理想隐藏状态过拟合。
- 搜索异常、搜索超时或收益不足时保留基础规则动作。

### 本地评测

- 每个提交在独立 Python 进程中导入和执行。
- 分别控制导入、读取牌组、单次行动、累计超额时间和整局运行时间。
- 非法动作、超时、崩溃和无结果均进入结构化统计。
- 支持双人多局对战和多提交 round-robin ladder。
- 使用 Kaggle 风格 TrueSkill 参数输出相对排名估计。
- 可按全部、失败、采样或关闭模式保存完整对局 trace。

### 自动优化

- 生成合法牌组变体、策略权重和完整 policy genome。
- 维护共同进化的 counter-opponent archive。
- 使用候选池稀疏比赛图代替昂贵的完整循环赛。
- 使用 Sequential Racing、稳健对局统计和角色分类筛选候选。
- 使用 PSRO 识别循环克制、脆弱候选和下一代重点对手。
- 从决定性失败中提取场景，驱动下一代 mutation pressure。
- 最终阶段使用独立 holdout 对手池，降低对发现池过拟合。

## Replay 驱动优化

最新一轮工作从真实 Kaggle episode 中提取失败模式，加入 Great Tusk 晚期自我牌库保护、按手牌资源动态选择首发、终盘 Crustle 墙路线，并修复本地评测把对手导入/牌组失败错误记为胜利的问题。原始日志和 replay 恢复出的竞争性资产不会提交到仓库。

完整诊断、修复清单、工具用法和验证原则见 `docs/KAGGLE_REPLAY_OPTIMIZATION.md`。

## 策略组合与历史表现

当前代码和已生成自有提交包含 **6 个核心策略家族、11 套去重牌表和 4 类主要提交逻辑**：

- **Great Tusk**：牌库破坏、资源干扰、Crustle 防守墙和应急 KO；
- **Mega Lucario**：快速进化、正面奖赏竞速和隐藏世界短程搜索；
- **Grimmsnarl**：进化攻击、伤害计数器和后备目标控制；
- **Mega Kangaskhan**：多属性能量、多攻击手和 Ogerpon 引擎；
- **Mega Lopunny**：Dudunsparce 循环、跨回合记忆和攻击窗口管理；
- **Teal Mask Ogerpon**：单教师回放克隆、Teal Dance 加速和多 Ogerpon 能量攻击。

提交逻辑分为原生规则、Meta Rules、Clone 和 Clone Search。Lopunny 的 `baseline`、`attack`、`cycle`、`full` 属于同一家族的 profile，不重复计为新策略。

截至 **2026-08-03** 的内部快照中，Great Tusk `gt_anti_crustle` 拥有当前最强可靠自有历史线上记录 `794.6`；Lopunny 在部分服务器 holdout 中表现突出但尚未在线验证；Ogerpon `deep_v2` 行为一致率达到 `94.90%`，而已提交的 `balanced_v2` Kaggle 分数为 `600.0`。这些结果同时说明本地胜率、行为一致率和 Kaggle 分数之间仍存在明显偏差。

详细牌表核心、教师来源、候选路径、优缺点、服务器评测、submission ID 和数据质量说明见内部文档 `docs/STRATEGY_PORTFOLIO.md`。

## 策略概览

当前默认牌组位于 `deck.csv`，包含 60 个卡牌 ID，每行一个整数。

主要战术组件：

- **Great Tusk**：使用 Land Collapse 进行牌库破坏，在需要时切换伤害输出。
- **Dwebble / Crustle**：建立防守墙，拖慢对方得分并保护牌库竞速计划。
- **Explorer's Guidance**：强化 Great Tusk 的牌库破坏回合。
- **Xerosic's Machinations、Boss's Orders 等**：控制双方资源与场面节奏。
- **Neutral Center 和工具牌**：针对 Pokémon ex 压力并提高生存能力。
- **Terrakion 等应急攻击手**：在纯牌库破坏无法安全获胜时提供 KO 路线。

`main.py` 会依据对手可见信息计算威胁，包括：

- Pokémon ex 或 ex 进化线压力；
- 即将形成的攻击能力；
- 特殊能量依赖；
- 可被 Boss 或控制效果捕获的后备 Pokémon；
- 双方剩余牌库与自身场面重建需求；
- 已知 archetype 特征及特定高风险对局。

## 架构

```text
.
├── main.py                     # Kaggle 单文件策略入口
├── deck.csv                    # 默认 60 张牌组
├── cg/                         # 官方 API、模拟器包装和多平台原生库
├── local_eval/                 # 隔离进程裁判、批量评测、排名和报告
├── tools/                      # 构建、搜索、发现、反馈和晋升工具
├── configs/                    # 榜分锚点等配置
├── tests/                      # 单元测试和评测系统回归测试
├── remote_*.sh                 # 长时间远程运行辅助脚本
├── ARCHITECTURE.md             # 模块依赖和维护约束
└── 基准/                        # 已跟踪的基准提交包
```

运行时分层：

1. `main.py` 提供 Kaggle 可调用策略。
2. `cg/` 包装官方模拟器、卡牌数据 API 和搜索接口。
3. `local_eval/` 在隔离进程中运行提交并生成比赛报告。
4. `tools/` 构建候选、执行搜索、分析失败并晋升冠军。
5. `tests/` 保护动作合法性、评测行为和生成流程。

工具模块的推荐依赖方向详见 `ARCHITECTURE.md`。新增自动化逻辑应放在 `tools/`，不要让 Kaggle 运行时意外依赖未打包模块。

## 运行要求

### Python

- 推荐 Python **3.10 或更高版本**。
- 核心代码主要使用 Python 标准库。
- 运行测试需要安装 `pytest`。

### 原生模拟器

仓库已包含以下原生运行库：

- Windows x86-64：`cg/cg.dll`
- Linux x86-64：`cg/libcg.so`
- Linux ARM64：`cg/libcg-arm64.so`
- macOS ARM64：`cg/libcg.dylib`

`cg/sim.py` 会根据操作系统和机器架构自动选择对应文件。如果动态库无法加载，请先确认操作系统、CPU 架构和文件执行权限是否匹配。

### 安装测试依赖

```bash
python3 -m pip install pytest
```

仓库目前没有强制虚拟环境方案。建议自行使用 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip pytest
```

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/NixumbraSolivagant/Pokemon-Agent.git
cd Pokemon-Agent
```

### 2. 验证环境

```bash
python3 -m compileall -q main.py cg local_eval tools tests
python3 -m pytest -q
```

### 3. 构建默认提交包

```bash
python3 -m tools.build_submission \
  --out outputs/submissions/champion_gt_search.tar.gz
```

构建器优先从 `--base` 压缩包读取牌组；如果基准包不存在或其中没有有效牌组，则回退到仓库根目录的 `deck.csv`。运行时的 `main.py` 与 `cg/` 始终使用当前仓库中的规范版本。

### 4. 导出 Kaggle 清洁包

```bash
python3 -m tools.export_kaggle_submission \
  outputs/submissions/champion_gt_search.tar.gz \
  --out outputs/submissions/submission.tar.gz
```

生产默认会保留搜索包装器，并移除 `build_metadata.json`、缓存文件和本地评测内部文件。

### 5. 与基准进行快速对战

```bash
python3 -m local_eval.cli match \
  outputs/submissions/submission.tar.gz \
  基准/submission_sorce_700.tar.gz \
  --games 20 \
  --workers 2
```

## 构建提交包

### 默认构建

```bash
python3 -m tools.build_submission
```

默认内部输出为：

```text
outputs/submissions/champion_gt_search.tar.gz
```

常用参数：

```bash
python3 -m tools.build_submission \
  --base 基准/submission_sorce_700.tar.gz \
  --out outputs/submissions/my_candidate.tar.gz \
  --name my_candidate \
  --enable-search \
  --search-candidates 8 \
  --search-budget-s 0.25 \
  --search-margin 1200 \
  --search-rollout-steps 16 \
  --belief-worlds 4 \
  --risk-penalty 0.20
```

关键选项：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--base` | 可选的基础提交压缩包 | `outputs/reference_submissions/i-have-one-rear-card.tar.gz` |
| `--out` | 内部候选输出路径 | `outputs/submissions/champion_gt_search.tar.gz` |
| `--injection` | 是否注入 Great Tusk 搜索包装器 | `great_tusk` |
| `--enable-search` | 启用原生短程搜索 | 开启 |
| `--search-candidates` | 搜索的候选动作数量 | `8` |
| `--search-budget-s` | 单次搜索预算，单位秒 | `0.25` |
| `--search-margin` | 覆盖规则动作所需的最低估值优势 | `1200` |
| `--search-rollout-steps` | 单条 rollout 最大步数 | `16` |
| `--belief-worlds` | 隐藏信息采样世界数量 | `4` |
| `--risk-penalty` | 多世界结果的风险惩罚 | `0.20` |

### 使用 JSON 构建配置

```json
{
  "name": "great_tusk_variant",
  "family": "great_tusk",
  "out": "outputs/submissions/great_tusk_variant.tar.gz",
  "enable_search": true,
  "search_candidates": 10,
  "search_budget_s": 0.30,
  "search_margin": 1000.0,
  "search_rollout_steps": 18,
  "belief_worlds": 6,
  "risk_penalty": 0.25,
  "deck_swaps": [[607, 58]],
  "policy_variant": "experimental",
  "notes": "Example experimental build"
}
```

保存为 `candidate.json` 后运行：

```bash
python3 -m tools.build_submission --config candidate.json
```

`deck_override` 也可以提供完整的 60 个卡牌 ID。构建器会验证牌组长度、卡牌 ID、同名卡数量和 ACE SPEC 等规则。

### 规则策略消融包

完全不注入搜索代码：

```bash
python3 -m tools.build_submission \
  --injection none \
  --out outputs/submissions/heuristic_only.tar.gz
```

从已有内部候选中移除搜索包装器：

```bash
python3 -m tools.export_kaggle_submission \
  outputs/submissions/champion_gt_search.tar.gz \
  --out outputs/submissions/heuristic_only_submission.tar.gz \
  --strip-search-wrapper
```

## 本地评测

### 两个提交多局对战

```bash
python3 -m local_eval.cli match \
  outputs/submissions/candidate_a.tar.gz \
  outputs/submissions/candidate_b.tar.gz \
  --games 100 \
  --workers 4 \
  --seed 20260717 \
  --record-mode losses
```

如果没有指定 `--out`，报告默认写入：

```text
reports/local_eval/run_YYYYMMDD_HHMMSS/
```

### Round-robin ladder

```bash
python3 -m local_eval.cli ladder \
  --submissions 'outputs/submissions/*.tar.gz' \
  --baseline 基准/submission_sorce_700.tar.gz \
  --games-per-pair 20 \
  --workers 4 \
  --record-mode sample \
  --record-sample-rate 0.05
```

### 评测限制参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--import-timeout` | 导入提交的超时秒数 | `6.0` |
| `--deck-timeout` | 获取牌组的超时秒数 | `6.0` |
| `--act-timeout` | 单次动作调用超时秒数 | `6.0` |
| `--overage-time` | 累计额外时间预算 | `12.0` |
| `--run-timeout` | 单局总运行超时秒数 | `1200.0` |
| `--max-actions` | 单局最大动作数量 | `1000` |
| `--workers` | 并行比赛线程数量 | `1` |

### Trace 保存模式

- `all`：记录所有对局。
- `losses`：保留决定性失败，适合后续失败挖掘。
- `sample`：按采样率保留对局。
- `none`：不保存详细 trace，节省磁盘和序列化开销。

使用 `--record-gzip` 可以压缩对局记录。长时间发现任务建议使用 `sample` 或 `losses`，避免输出目录快速膨胀。

## 自动策略发现

自动发现入口为：

```bash
python3 -m tools.discovery_engine
```

可用子命令：

- `run`：运行多阶段策略发现。
- `status`：查看当前发现状态。
- `audit`：重新分类最终结果并生成决策简报。
- `mine-losses`：从对局记录提取失败场景。
- `digest-losses`：将失败场景聚合为下一代压力摘要。
- `run-scenarios`：对候选重放关键场景。
- `export-final`：重新导出已晋升的最终提交。

### Smoke 验证

首次运行建议先使用小规模 profile：

```bash
python3 -m tools.discovery_engine run \
  --profile smoke_discovery \
  --out outputs/discovery_smoke \
  --incumbent outputs/submissions/champion_gt_search.tar.gz \
  --pool 基准/submission_sorce_700.tar.gz \
  --progress
```

`smoke_discovery` 只用于验证构建、评测和晋升链路，不应把少量对局结果视为可靠策略结论。

### 标准发现

```bash
python3 -m tools.discovery_engine run \
  --profile a800_discovery \
  --out outputs/discovery_a800 \
  --incumbent outputs/submissions/champion_gt_search.tar.gz \
  --pool \
    基准/submission_sorce_700.tar.gz \
    outputs/reference_submissions/submission_820.tar.gz \
    outputs/reference_submissions/multiply-agent-best-940-lb.tar.gz \
  --workers 0 \
  --progress
```

`workers=0` 表示根据 CPU 数量和 headroom 自动确定并发数。可使用 `--max-workers` 和 `--cpu-headroom` 控制资源上限。

### 发现 Profiles

| Profile | 用途 | 默认代数 | 种群 | Stage D 每对局数 |
| --- | --- | ---: | ---: | ---: |
| `smoke_discovery` | 流程验证 | 1 | 10 | 1 |
| `a800_discovery` | 标准高置信度发现 | 24 | 144 | 512 |
| `a800_turbo_discovery` | 更宽候选和对手池 | 24 | 144 | 512 |

标准 profile 计算量很高。Stage D 会对最终候选进行高局数交叉验证，并且完整运行可能产生大量压缩包、JSON 报告和 trace。

### 查看状态与审计

```bash
python3 -m tools.discovery_engine status \
  --out outputs/discovery_a800
```

```bash
python3 -m tools.discovery_engine audit \
  --out outputs/discovery_a800
```

### 导出最终候选

```bash
python3 -m tools.discovery_engine export-final \
  --out outputs/discovery_a800 \
  --submission-out outputs/submissions/submission.tar.gz
```

## 其他优化工作流

### 低资源自动迭代

```bash
python3 -m tools.auto_iterate \
  --generations 3 \
  --population 24 \
  --stage1-games 4 \
  --stage2-games 16 \
  --finalists 4 \
  --workers 4
```

适合在有限计算资源下快速比较候选。它仍然使用晋升阈值和无结果率约束，但置信度低于完整 discovery profile。

### 高置信度 Gold Sprint

```bash
python3 -m tools.gold_sprint \
  --cycles 3 \
  --generations-per-cycle 4 \
  --population 96 \
  --workers 8 \
  --pool 基准/submission_sorce_700.tar.gz
```

Gold Sprint 用于无人值守的高置信度候选迭代，包含多阶段筛选、最终验证、冠军晋升和状态恢复。

### Gold Factory

```bash
python3 -m tools.gold_factory --help
```

Gold Factory 提供另一套候选生成与评测流程，可生成跨 archetype seed、搜索参数变体和牌组 mutation。

### 失败挖掘

完整发现流程会自动处理失败反馈，也可以手动运行：

```bash
python3 -m tools.discovery_engine mine-losses --help
python3 -m tools.discovery_engine digest-losses --help
python3 -m tools.discovery_engine run-scenarios --help
```

失败摘要用于识别：

- 早期场面崩溃；
- 贴能或撤退路线错误；
- 对特定 archetype 的系统性弱点；
- 牌库、手牌或奖赏竞速失败；
- 候选只在平均结果上较好、但存在严重尾部风险的情况。

## 牌组与提交格式

### `deck.csv`

- 必须恰好包含 60 行有效整数。
- 每行是一个卡牌 ID。
- 构建和本地裁判会验证基础牌组约束。
- 不要添加表头、逗号或注释。

示例：

```text
58
58
344
344
...
```

### 提交压缩包

本地评测要求 `.tar.gz` 至少包含：

```text
main.py
deck.csv
cg/
cg/game.py
cg/api.py
```

构建器还会打包当前 `cg/` 中对应平台的原生库。内部候选通常包含 `build_metadata.json`，Kaggle 清洁导出默认移除该文件。

压缩包解压使用安全路径检查，绝对路径、目录穿越、符号链接和硬链接会被拒绝。

## 测试与验证

推荐验证顺序：

```bash
python3 -m compileall -q main.py cg local_eval tools tests
git diff --check
python3 -m pytest -q
```

当前测试覆盖：

- `deck.csv` 路径解析和基础 Agent fallback；
- 动作边界、重复选择和牌组合法性；
- 安全解压和路径穿越防护；
- 提交导入、读取牌组和执行动作的超时处理；
- Agent stdout 噪声隔离；
- TrueSkill 更新和评测报告；
- 失败 trace 记录与候选比赛调度；
- 搜索构建注入幂等性；
- Kaggle 清洁包导出；
- policy genome、交叉、mutation 和策略 archive；
- reference-free 候选生成及 counter opponent；
- discovery profile、反馈压力、PSRO 和 gold gate。

只运行核心测试：

```bash
python3 -m pytest -q tests/test_main.py tests/local_eval/test_core.py
```

只运行发现系统测试：

```bash
python3 -m pytest -q \
  tests/test_reference_free_search.py \
  tests/test_policy_genome.py \
  tests/local_eval/test_discovery_engine.py
```

## 输出文件

以下目录默认被 `.gitignore` 排除：

- `outputs/`：候选提交、发现状态、晋升结果和参考提交。
- `reports/`：本地评测报告和对局记录。
- `log/`：运行日志。
- `参考/`：本地研究 Notebook。
- `pokemon-tcg-ai-battle/`：本地官方或第三方参考工程。

常见输出：

```text
outputs/submissions/*.tar.gz
outputs/discovery_*/state.json
outputs/discovery_*/generation_*/report.json
outputs/discovery_*/generation_*/decision.json
outputs/discovery_*/decision_brief.md
reports/local_eval/run_*/report.json
reports/local_eval/run_*/game_records/
```

建议定期归档或删除失败候选的压缩包和高体积 trace。生产级发现任务应优先保留：

- 最终候选；
- incumbent 和 Hall of Fame；
- 决策 JSON 与简报；
- 关键失败场景；
- 可复现所需的种子、配置和 SHA256。

## 远程运行脚本

仓库提供多套 Shell 脚本用于远程服务器或长时间任务：

- `remote_run.sh`：通用自动迭代入口。
- `remote_gold_run.sh`：Gold 运行入口。
- `remote_gold_factory.sh`：Gold Factory 入口。
- `remote_gold_sprint.sh`：高置信度 Gold Sprint。
- `remote_gold_discovery.sh`：完整 discovery engine。

运行前应先阅读脚本中的默认路径、CPU 数量、运行周期和对手池配置：

```bash
sed -n '1,240p' remote_gold_discovery.sh
```

然后根据机器资源调整并执行：

```bash
./remote_gold_discovery.sh
```

这些脚本可能启动高计算量和高磁盘占用任务，不建议在未检查参数时直接运行。

## 开发约定

- 保持 `main.py` 为 Kaggle 可用的单文件运行入口。
- 卡牌 ID 和直接比赛启发式暂时保留在 `main.py`，直到引入可靠的源码到单文件构建步骤。
- 公开 Agent 失败必须通过统一 fallback 处理。
- 所有随包文件必须相对于 `__file__` 解析，不能假设当前工作目录。
- 搜索包装器属于生成代码；其模板行为应在 `tools/build_submission.py` 中维护。
- 新工具代码应从其真正拥有数据或规则的模块导入，避免循环依赖。
- 不要把 reference submission 当作候选源码复制来源。
- 修改 archive 构建或搜索注入逻辑时，应增加压缩包级回归测试。
- 不要提交 `outputs/`、评测 trace、缓存和本地参考 Notebook。

更完整的模块边界说明见 `ARCHITECTURE.md`。

## 常见问题

### `OSError: cannot open shared object file`

确认 `cg/` 下存在与你的平台和架构对应的原生库，并确认仓库文件没有在下载或解压时损坏。

Linux 可检查：

```bash
file cg/libcg.so
ldd cg/libcg.so
```

### 默认 `--base` 文件不存在

`outputs/reference_submissions/` 不进入 Git。可以显式指定仓库自带基准：

```bash
python3 -m tools.build_submission \
  --base 基准/submission_sorce_700.tar.gz
```

也可以不提供有效基准包；构建器会使用根目录 `deck.csv`，运行时仍来自当前 `main.py` 和 `cg/`。

### 本地比赛大量超时

- 降低 `--workers`；
- 提高 `--run-timeout`；
- 确认机器没有发生 CPU 严重过载；
- 减少搜索候选数、rollout 步数或搜索预算；
- 使用 `--record-mode none` 排除 trace 序列化开销。

### 输出目录增长过快

- 使用 `--record-mode losses` 或 `sample`；
- 降低 `--record-sample-rate`；
- 对 game records 启用 gzip；
- 定期只保留冠军、关键失败和最终报告。

### 本地高分是否等于 Kaggle 榜分

不等于。本地 TrueSkill 结果只表示当前模拟器、种子、对手池和时间限制下的相对表现。可靠晋升应同时满足：

- 对 incumbent 的稳定优势；
- 对多个 archetype 的清洁结果；
- 足够低的超时、崩溃和无结果率；
- 独立 holdout 对手池表现；
- 足够大的对局样本；
- 必要时进行 Kaggle 实际提交验证。

### 如何查看所有 CLI 参数

```bash
python3 -m local_eval.cli --help
python3 -m local_eval.cli match --help
python3 -m local_eval.cli ladder --help
python3 -m tools.build_submission --help
python3 -m tools.discovery_engine --help
python3 -m tools.discovery_engine run --help
python3 -m tools.auto_iterate --help
python3 -m tools.gold_factory --help
python3 -m tools.gold_sprint --help
```

## 仓库状态说明

- GitHub 仓库：`NixumbraSolivagant/Pokemon-Agent`
- 主分支：`master`
- 本仓库未跟踪大型实验产物、参考 Notebook 和本地第三方工程。
- `configs/lb_anchor_table.json` 中的榜分锚点用于内部评测目标，不代表官方保证或实时排行榜结果。
- 仓库当前未提供独立许可证文件；复用代码或原生模拟器文件前，请确认相关比赛规则和上游资源许可。

---

如果只想运行现有策略，依次执行“构建提交包 → 导出 Kaggle 清洁包 → 本地对战”即可。如果要继续提高策略强度，建议先运行 `smoke_discovery` 验证环境，再启动标准 discovery 或 Gold Sprint。
