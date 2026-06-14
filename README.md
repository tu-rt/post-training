# Post-Training: Qwen 中文指令 LoRA 微调与评测

基于 Qwen2.5 系列模型，使用 COIG-CQIA 中文指令数据集进行 LoRA/QLoRA 监督微调（SFT），并通过 C-Eval 客观评测与 Hold-out 主观评测量化微调效果，同时完成数据规模消融实验。

## 技术栈

- **基座模型**: Qwen2.5-0.5B / 1.5B / 7B-Instruct
- **微调方法**: LoRA / QLoRA (PEFT + bitsandbytes)
- **训练框架**: HuggingFace Transformers + TRL SFTTrainer
- **训练数据**: [BAAI/COIG-CQIA](https://huggingface.co/datasets/BAAI/COIG-CQIA)（7 个子集：zhihu, wiki, exam, xhs, douban, wikihow, coig_pc）
- **客观评测**: C-Eval（5 科：计算机网络、操作系统、离散数学、概率统计、大学化学）
- **主观评测**: Hold-out 集 Win-rate（Base vs LoRA）

## 实验设计

| 实验ID | 说明 |
|--------|------|
| E0 | Base 模型直接评测（对照基线） |
| E1 | LoRA SFT，小数据量（200 条） |
| E2 | LoRA SFT，中数据量（1K 条） |
| E3 | LoRA SFT，大数据量（5K / 10K 条，需 4090） |

核心消融维度：**训练数据规模对微调效果的影响**。

## 关键实现

- **4bit QLoRA 训练**: NF4 量化 + 双量化 + 梯度检查点，单卡 RTX 3060 (4GB) 即可训练 1.5B 模型
- **数据清洗管线**: 自动下载 COIG-CQIA → 多字段兼容抽取（instruction/output, input/response, conversation 等格式）→ 去重 → 长度过滤 → 随机打乱 → JSONL 缓存
- **双维度评测**: C-Eval 多选题准确率（客观）+ Hold-out 集 Win-rate（主观），避免单一指标偏差
- **Profile 配置系统**: `config.yaml` 支持 local/lab 双 profile 切换，无需改代码即可适配不同硬件

## 项目结构

```
post-training/
├── config.yaml              # 双 profile 配置（local 4GB / lab 4090）
├── requirements.txt
├── scripts/
│   ├── prepare_data.py      # 数据下载、清洗、划分
│   ├── train_lora.py        # LoRA 微调训练
│   ├── evaluate_ceval.py    # C-Eval 客观评测
│   ├── evaluate_holdout.py  # Hold-out 主观评测
│   ├── run_ablation.py      # 消融实验一键运行
│   ├── run_overnight.py     # 实验室长时间训练脚本
│   ├── run_recover.py       # 断点续训
│   ├── smoke_test.py        # 快速冒烟测试
│   └── preflight_check.py   # 训练前环境检查
├── src/
│   ├── model_utils.py       # 模型加载 / LoRA 构建 / 推理生成
│   ├── data_utils.py        # COIG-CQIA 数据加载与清洗
│   ├── config.py            # Profile 配置解析
│   ├── adapter_utils.py     # Adapter 路径管理
│   └── eval_ceval.py        # C-Eval 评测核心逻辑
├── data/                    # 自动生成（已 gitignore）
├── outputs/                 # LoRA Adapter 输出（已 gitignore）
└── results/                 # 评测结果 CSV（已 gitignore）
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

### 冒烟测试（推荐首次运行）

```bash
python scripts/smoke_test.py
```

流程：准备 200 条数据 → 训练 30 step → C-Eval + Hold-out 评测（5 条）。

### 分步运行

```bash
# 1. 准备数据
python scripts/prepare_data.py --train_size 200 --holdout_size 20

# 2. 训练 LoRA
python scripts/train_lora.py --train_size 200 --exp_id E1

# 3. 评测 Base 模型
python scripts/evaluate_ceval.py --mode base --exp_id E0

# 4. 评测 LoRA 模型
python scripts/evaluate_ceval.py --mode lora --adapter outputs/E1_size200_lr2e-4/adapter --exp_id E1
python scripts/evaluate_holdout.py --adapter outputs/E1_size200_lr2e-4/adapter --exp_id E1
```

### 完整消融实验

```bash
python scripts/run_ablation.py --steps prepare,train,eval
```

结果汇总：`results/ablation_summary.csv`

## 硬件适配

| 配置项 | Local (RTX 3060 4GB) | Lab (RTX 4090 24GB) |
|--------|----------------------|---------------------|
| Profile | `active_profile: local` | `active_profile: lab` |
| 基座模型 | Qwen2.5-1.5B 4bit | Qwen2.5-7B 4bit |
| LoRA rank | 8 | 16 |
| Target modules | q/k/v/o_proj | q/k/v/o/gate/up/down_proj |
| 训练数据量 | 200 / 1K | 1K / 5K / 10K |
| max_seq_length | 1024 | 2048 |

切换方式：修改 `config.yaml` 中 `active_profile` 即可，代码无需改动。

## 4GB 显存优化策略

1. QLoRA 4bit 量化 + 梯度检查点，1.5B 模型仅占约 2GB 显存
2. `per_device_train_batch_size=1` + `gradient_accumulation_steps=16` 等效 batch=16
3. 嵌入模型在 CPU 运行，不占用 GPU 显存
4. 0.5B 模型作为 bitsandbytes 兼容性备选
