# 使用 LLaMA-Factory 微调 Xing4.0-29B-A4B 模型

　　本文介绍如何在 LLaMA-Factory 中适配、加载、微调 Xing4.0-29B-A4B 模型。主要覆盖以下内容：

- LLaMA-Factory 环境安装与校验
- Xing4.0-29B-A4B 模型下载与目录准备
- 训练数据准备
- 支持使用全参、LoRA / QLoRA 进行 SFT 微调

## 环境准备

　　建议使用 Python 3.11 及以上版本。截止文档书写当日 LLaMA-Factory 仓库的 `pyproject.toml` 中要求：

```text
python >= 3.11
torch >= 2.4.0
transformers >= 4.55.0
```

　　安装 LLaMA-Factory：

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

　　或者从暂未合入的PR安装 LLaMA-Factory，在 Xing4.0 支持合入 LLaMA-Factory 主线前，请使用 PR #10754 分支；合入后可直接安装主线版本：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

git fetch origin pull/10754/head:pr-10754
git checkout pr-10754

pip install -e .
pip install -r requirements/metrics.txt
```

　　安装完成后，可以通过以下命令校验：

```bash
llamafactory-cli version
```

　　如需使用 DeepSpeed、FlashAttention-2 或 bitsandbytes，可根据训练环境按需安装：

```bash
pip install deepspeed
pip install flash-attn --no-build-isolation
pip install bitsandbytes
```

　　使用 FlashAttention-2 时，请确认 CUDA、PyTorch 与 flash-attn 版本兼容。

## 下载 Xing4.0-29B-A4B 模型

　　Xing4.0-29B-A4B 在 LLaMA-Factory 中注册的模型名为：

```text
XingChen-AGI/Xing4.0-29B-A4B
```

　　可以通过 ModelScope 下载到本地目录：

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.0-29B-A4B \
  --local_dir /yourpath/models/Xing4.0-29B-A4B
```

　　也可以使用已经准备好的 Hugging Face 格式模型目录，例如：

```text
/yourpath/models/Xing4.0-29B-A4B
```

## 数据准备

　　LLaMA-Factory 支持多种数据格式。Xing4.0-29B-A4B 的训练数据可以继续使用 LLaMA-Factory 标准的 alpaca 或 sharegpt 格式。

### Alpaca 格式

　　将数据保存为 JSON 文件，例如 `data/xing4.0_29b_a4b_sft.json`：

```json
[
  {
    "instruction": "请介绍 Xing4.0-29B-A4B 的主要特点。",
    "input": "",
    "output": "Xing4.0-29B-A4B 是面向通用对话和复杂任务的大语言模型，支持长上下文、指令跟随和多轮交互。",
    "system": "你是中国电信星辰语义大模型 Xing4.0-29B-A4B。"
  }
]
```

　　然后在 `data/dataset_info.json` 中注册：

```json
{
  "xing4.0_29b_a4b_sft": {
    "file_name": "xing4.0_29b_a4b_sft.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
}
```

### ShareGPT 格式

　　如果使用多轮对话数据，可以组织为 sharegpt 格式：

```json
[
  {
    "conversations": [
      {
        "from": "user",
        "value": "请介绍一下你自己。"
      },
      {
        "from": "assistant",
        "value": "我是 Xing4.0-29B-A4B，由中电信人工智能科技有限公司研发的大语言模型。"
      }
    ],
    "system": "你是中国电信星辰语义大模型 Xing4.0-29B-A4B。"
  }
]
```

　　对应的 `dataset_info.json` 配置示例：

```json
{
  "xing4.0_29b_a4b_sharegpt": {
    "file_name": "xing4.0_29b_a4b_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

　　如果数据中的角色值是 `human` / `gpt`，需要将 `user_tag` / `assistant_tag` 分别改为 `human` / `gpt`。

　　

## SFT 微调

### LoRA / QLoRA SFT 微调

　　可以在当前仓库中新建训练配置文件：

```text
examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml
```

　　配置内容如下：

```yaml
### model
model_name_or_path: /yourpath/models/Xing4.0-29B-A4B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: xing4.0
cutoff_len: 1024
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/xing4.0-29b-a4b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### qlora
quantization_bit: 4
optim: paged_adamw_8bit
```

　　需要根据实际模型路径修改 `model_name_or_path`：

```yaml
model_name_or_path: /yourpath/models/Xing4.0-29B-A4B
```

　　当前示例配置中启用了：

```yaml
quantization_bit: 4
```

　　因此该示例实际为 4-bit QLoRA SFT。如果希望执行普通 LoRA，请删除或注释 `quantization_bit: 4`，并根据显存情况调整 batch size、cutoff length 和优化器等。

### 单卡训练

　　在 LLaMA-Factory 根目录下执行：

```bash
llamafactory-cli train examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml
```

　　如需指定单张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 \
llamafactory-cli train examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml
```

需根据显存调整 `cutoff_len`、batch size、量化配置；若 OOM，优先使用多卡/DeepSpeed。

### 多卡训练

　　LLaMA-Factory 会默认使用所有可见 GPU。也可以显式指定多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FORCE_TORCHRUN=1 \
llamafactory-cli train examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml
```

　　如果使用 DeepSpeed ZeRO-3，可在 YAML 中增加：

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

　　然后执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FORCE_TORCHRUN=1 \
llamafactory-cli train examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml
```

### 覆盖 YAML 参数

　　LLaMA-Factory 支持在命令行覆盖 YAML 中的参数。例如调整学习率和日志间隔：

```bash
llamafactory-cli train examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml \
  learning_rate=5e-5 \
  logging_steps=1
```

### 全参 SFT 微调

全参微调会更新模型全部参数，显存与通信开销显著高于 LoRA / QLoRA，建议在多卡环境中配合 DeepSpeed 使用。如需降低显存占用，可优先调整 `cutoff_len`、`per_device_train_batch_size`、`gradient_accumulation_steps`，或使用 ZeRO-2 / ZeRO-3。

　　可以新建全参训练配置文件，例如：

```text
examples/train_full/xing4.0_29b_a4b_full_sft.yaml
```

　　相比 LoRA / QLoRA，全参微调的关键差异如下：

```yaml
model_name_or_path: /yourpath/models/Xing4.0-29B-A4B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

dataset: xing4.0_29b_a4b_sft
template: xing4.0
cutoff_len: 1024

output_dir: saves/xing4.0-29b-a4b/full/sft
save_only_model: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 1.0
bf16: true
```

　　多卡启动命令示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FORCE_TORCHRUN=1 \
llamafactory-cli train examples/train_full/xing4.0_29b_a4b_full_sft.yaml
```

　　如果使用上文的 ShareGPT 数据，请将 `dataset` 修改为 `xing4.0_29b_a4b_sharegpt`。正式训练前建议先使用少量样本完成 smoke test，确认模型加载、数据模板、显存占用和 loss 输出均正常。

### 恢复训练

　　如果训练中断，可将 `resume_from_checkpoint` 指向 checkpoint 目录：

```bash
llamafactory-cli train examples/train_lora/xing4.0_29b_a4b_lora_sft.yaml \
  resume_from_checkpoint=saves/xing4.0-29b-a4b/lora/sft/checkpoint-500
```
