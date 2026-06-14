# Post-Training: Qwen 中文指令 LoRA 微调与评测

本项目基于 Qwen2.5 系列模型，使用 COIG-CQIA 中文指令数据进行 LoRA/QLoRA 监督微调（SFT），并通过 C-Eval 客观评测与 Hold-out 自动对比评测量化微调效果。项目重点关注小规模中文指令数据在不同训练规模下的微调收益与稳定性。

## 实验结果速览

| 实验 | 训练数据量 | C-Eval 准确率 | Hold-out win-rate | 说明 |
|------|------------|---------------|-------------------|------|
| E0 | 0 | 20% | - | Base 模型直接评测 |
| E2 | 1,000 | 24% | 55% | 当前最优性价比配置 |
| E3 | 5,000 | 指标下降 | 可用但收益不稳定 | 出现过拟合/泛化下降迹象 |
| E4 | 10,000 | 模式崩溃 | 不稳定 | 当前超参下不适合作为最终配置 |

> 注：Hold-out win-rate 为自动对比指标，基于模型输出与参考答案的 overlap_score 计算，用于辅助比较 Base 与 LoRA，不等同于人工主观评分。

## 技术栈

- **基座模型**: Qwen2.5-0.5B / 1.5B / 7B-Instruct
- **微调方法**: LoRA / QLoRA（PEFT + bitsandbytes）
- **训练框架**: HuggingFace Transformers + TRL SFTTrainer
- **训练数据**: [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA)
- **客观评测**: C-Eval（计算机网络、操作系统、离散数学、概率统计、大学化学）
- **自动对比评测**: Hold-out 集 Base vs LoRA win-rate

## 实验设计

| 实验ID | 说明 |
|--------|------|
| E0 | Base 模型直接评测 |
| E1 | LoRA SFT，小数据量（local: 200 条） |
| E2 | LoRA SFT，中数据量（lab: 1K 条） |
| E3 | LoRA SFT，大数据量（lab: 5K / 10K 条） |

核心消融维度：**训练数据规模对 QLoRA 指令微调效果的影响**。

## 关键实现

- **4bit QLoRA 训练**: NF4 量化 + 双量化 + 梯度检查点；本机 4GB 显存可训练 1.5B，实验室 4090 可训练 7B
- **单卡训练约束**: 训练脚本默认限制 `CUDA_VISIBLE_DEVICES=0`，避免 7B QLoRA 在 `device_map="auto"` 多卡分片时出现跨设备 loss 问题
- **数据清洗管线**: 自动下载 COIG-CQIA → 多字段兼容抽取 → 去重 → 长度过滤 → 随机打乱 → JSONL 缓存
- **双轨评测**: C-Eval 多选题准确率 + Hold-out 自动对比 win-rate，避免只看单一指标
- **Profile 配置系统**: `config.yaml` 支持 `local/lab` 双 profile，便于在本机与实验室服务器之间切换

## 项目结构

```text
post-training/
├── config.yaml              # local/lab 双 profile 配置
├── requirements.txt
├── scripts/
│   ├── prepare_data.py      # 数据下载、清洗、划分
│   ├── train_lora.py        # LoRA / QLoRA 微调
│   ├── evaluate_ceval.py    # C-Eval 客观评测
│   ├── evaluate_holdout.py  # Hold-out 自动对比评测
│   ├── run_ablation.py      # 消融实验一键运行
│   ├── smoke_test.py        # 快速冒烟测试
│   ├── run_overnight.py     # 实验室长时间训练脚本
│   ├── run_recover.py       # 断点续训/恢复
│   └── preflight_check.py   # 训练前环境检查
├── src/
│   ├── model_utils.py
│   ├── data_utils.py
│   ├── config.py
│   ├── adapter_utils.py
│   └── eval_ceval.py
├── data/                    # 自动生成，已 gitignore
├── outputs/                 # LoRA Adapter，已 gitignore
└── results/                 # 评测结果，已 gitignore
```

## 快速开始

### 环境准备

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 国内用户建议设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

需要 CUDA 版 PyTorch。

### 冒烟测试

```bash
python scripts/smoke_test.py
```

流程：准备 200 条数据 → 训练 30 step → C-Eval + Hold-out 小样本评测。

### 分步运行

```bash
# 1. 准备数据
python scripts/prepare_data.py --train_size 200 --holdout_size 20

# 2. 训练 LoRA
python scripts/train_lora.py --train_size 200 --exp_id E1

# 3. 评测 Base
python scripts/evaluate_ceval.py --mode base --exp_id E0

# 4. 评测 LoRA
python scripts/evaluate_ceval.py --mode lora --adapter outputs/E1_size200_lr2e-4/adapter --exp_id E1
python scripts/evaluate_holdout.py --adapter outputs/E1_size200_lr2e-4/adapter --exp_id E1
```

### 完整消融实验

```bash
python scripts/run_ablation.py --steps prepare,train,eval
```

结果汇总输出到 `results/ablation_summary.csv`。

## 硬件适配

| 配置项 | Local (RTX 3060 4GB) | Lab (RTX 4090 24GB) |
|--------|----------------------|---------------------|
| Profile | `active_profile: local` | `active_profile: lab` |
| 基座模型 | Qwen2.5-1.5B 4bit | Qwen2.5-7B 4bit |
| LoRA rank | 8 | 16 |
| Target modules | q/k/v/o_proj | q/k/v/o/gate/up/down_proj |
| 训练数据量 | 200 / 1K | 1K / 5K / 10K |
| max_seq_length | 1024 | 2048 |

公开仓库建议默认使用 `active_profile: local`。如果在实验室服务器复现实验，需要将 `lab.model.name` 改成本地模型路径或 HuggingFace 模型 ID。

## 评测指标说明

- **C-Eval accuracy**：模型对单选题选项 A/B/C/D 的准确率。
- **Hold-out score**：基于模型输出与参考答案的字符级 overlap_score。
- **Hold-out win-rate**：LoRA 的 overlap_score 高于 Base 的样本比例。

该 Hold-out 评测是轻量自动评测，主要用于比较不同微调配置的相对变化，不等同于人工偏好评测或 LLM-as-judge。

## 主要结论

- 小规模高质量中文指令数据对 Qwen2.5-7B 的部分任务表现有提升。
- 1K 数据配置在当前实验中取得最佳性价比：C-Eval +4pt，Hold-out win-rate 55%。
- 5K / 10K 数据在当前学习率和 epoch 配置下未持续提升，提示 SFT 并非简单增加数据量即可变好。
- 后续可继续从数据质量筛选、学习率缩放、epoch 调整和偏好优化（DPO）方向改进。
