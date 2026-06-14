# 中文指令 LoRA 微调与评测（低显存版）

面向 **RTX 3060 Laptop 4GB** 的本机可跑 **项目一：后训练 / SFT**，与 `RAG 应用算法` 项目配套。实验室 **4090** 只需改 `config.yaml` 中 `active_profile: lab`。

## 功能

| 实验 | 说明 |
|------|------|
| **E0** | 仅评测 base 模型（对照） |
| **E1** | LoRA，较小数据量（local: 200 条） |
| **E2** | LoRA，较大数据量（local: 1k 条） |

- 训练数据：[BAAI/COIG-CQIA](https://huggingface.co/datasets/BAAI/COIG-CQIA)（自动下载 + 去重清洗）
- 客观评测：C-Eval 子集（5 科 × N 题）
- 主观评测：Hold-out 集 base vs LoRA win-rate

## 环境准备

```powershell
cd "C:\Users\hasee\Desktop\Cursor\后训练算法"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

需 **CUDA 版 PyTorch**（与项目二相同环境可复用）。

### bitsandbytes 失败时

编辑 `config.yaml` → `profiles.local.model`：

```yaml
name: "Qwen/Qwen2.5-0.5B-Instruct"
use_4bit: false
```

## 快速运行

### 1）一键冒烟（推荐首次）

```powershell
python scripts\smoke_test.py
# 或双击 run_smoke.bat
```

流程：准备 200 条数据 → 训练 30 step → C-Eval + Hold-out（5 条）。

### 2）分步运行

```powershell
# 准备数据（local 默认最大 1000 + holdout 50）
python scripts\prepare_data.py --train_size 200 --holdout_size 20

# 训练（本机建议先 200 条）
python scripts\train_lora.py --train_size 200 --exp_id E1

# 评测 base
python scripts\evaluate_ceval.py --mode base --exp_id E0

# 评测 LoRA（路径按实际 outputs 目录修改）
python scripts\evaluate_ceval.py --mode lora --adapter outputs\E1_size200_lr2e-4\adapter --exp_id E1

python scripts\evaluate_holdout.py --adapter outputs\E1_size200_lr2e-4\adapter --exp_id E1
```

### 3）本机完整消融

```powershell
python scripts\run_ablation.py --steps prepare,train,eval
```

结果汇总：`results/ablation_summary.csv`

## 目录结构

```text
后训练算法/
├── config.yaml              # local / lab 两套 profile
├── scripts/
│   ├── prepare_data.py
│   ├── train_lora.py
│   ├── evaluate_ceval.py
│   ├── evaluate_holdout.py
│   ├── run_ablation.py
│   └── smoke_test.py
├── src/
├── data/                    # 自动生成
├── outputs/                 # LoRA adapter
└── results/                 # 评测 csv
```

## 本机 vs 实验室配置

| 项 | local（4GB） | lab（4090） |
|----|--------------|-------------|
| 切换方式 | `active_profile: local` | `active_profile: lab` |
| 模型 | Qwen2.5-**1.5B** 4bit | Qwen2.5-**7B** 4bit |
| 训练量 | 200 / 1k | 1k / 5k / 10k |
| C-Eval | 5×10=50 题 | 5×20=100 题 |
| max_seq_length | 1024 | 2048 |

实验室步骤：

```powershell
# 1. 改 config.yaml: active_profile: lab
# 2. 同样命令跑 ablation
python scripts\run_ablation.py --steps prepare,train,eval
```

RAG 项目（`RAG 应用算法`）同步改 `active_profile` 或手动把 `llm` 改为 7B 即可。

## 显存建议（4GB）

1. 关闭占用 GPU 的其他程序  
2. 仅用 **1.5B + QLoRA**，不要在本机试 7B  
3. `per_device_train_batch_size=1`，靠 gradient_accumulation  
4. 首次用 `--max_steps 30` 冒烟再跑完整 epoch  

## 简历指标（跑完后填入）

- 基于 Qwen2.5 + COIG-CQIA **N 条** LoRA SFT  
- C-Eval 子集准确率：base **x%** → LoRA **y%**（+z pt）  
- Hold-out win-rate **w%**  
- 数据规模消融（200 vs 1k）  

## 常见问题

**Q: COIG-CQIA 下载慢**  
设 `HF_ENDPOINT=https://hf-mirror.com`，第二次会用 `data/coig_cqia_clean.json` 缓存。

**Q: CUDA OOM**  
减小 `train_size`；`max_seq_length` 改为 512；用 0.5B 模型。

**Q: trl 版本报错 `processing_class`**  
`pip install trl>=0.9.6`；若仍失败可降级到 0.8 并把 `processing_class` 改为 `tokenizer`。

**Q: adapter 路径找不到**  
查看 `outputs/` 下文件夹名（含 `lr2e-4`），与命令中 `--adapter` 保持一致。

## 与项目二联调（实验室）

1. 本机两项目均 `smoke_test` 通过  
2. 实验室：`config.yaml` 均切到 **lab / 7B**  
3. 先跑完 **后训练 ablation**，再跑 **RAG evaluate**（可并行占不同 GPU）  
4. 汇总 `results/ablation_summary.csv` 与 `RAG/results/metrics_summary.csv` 写简历  
