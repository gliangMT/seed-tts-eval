# Seed-TTS-Eval + CosyVoice3 + MUSA 评测入门指南

本文档说明如何在摩尔线程 MUSA 环境中，使用 `seed-tts-eval` 评测
CosyVoice3 基础模型或微调 checkpoint。

本文以英文 Seed-TTS 测试集为主，完整流程分为三步：

1. 使用 CosyVoice3 生成测试集语音。
2. 使用 Whisper-large-v3 计算 WER。
3. 使用 WavLM-Large 计算说话人相似度 SIM。

本文档按当前仓库脚本编写。开始前需要知道三个边界：

- 普通全量评测包含推理、WER 和 SIM。
- `eval_cosyvoice_checkpoint.sh` 与 `watch_cosyvoice_checkpoints.sh` 当前只自动计算
  推理和英文 WER，不自动计算 SIM。
- 脚本结束或出现 `.complete` 不等于所有样本都成功，最终必须核对 wav 数量和
  指标文件中的有效样本数。

当前工作目录约定如下：

```text
/home/cosyvoice-test/
├── CosyVoice/
├── seed-tts-eval/
├── data/
│   ├── seedtts_testset/
│   └── models/
├── outputs/
│   └── seedtts_eval/
└── pretrained_models/
    ├── Fun-CosyVoice3-0.5B/
    └── Fun-CosyVoice3-0.5B-test/
```

## 1. 先理解各组件负责什么

### 1.1 CosyVoice3

CosyVoice3 负责根据以下三项内容生成语音：

- 要合成的文本 `infer_text`
- 提示语音对应的文本 `prompt_text`
- 提示语音 `prompt_wav`

Seed-TTS 的 `meta.lst` 正好提供这些字段，所以可以用
`inference_zero_shot()` 批量生成测试音频。

### 1.2 seed-tts-eval

原始 `seed-tts-eval` 主要负责指标计算，不负责训练模型。当前仓库在此基础上
增加了 CosyVoice 推理和 checkpoint 自动评测脚本，因此现在可以完成：

- 组织 Seed-TTS 测试集
- 使用 CosyVoice 生成 Seed-TTS 音频
- 使用 ASR 识别生成语音并计算 WER
- 使用说话人模型比较生成语音和提示语音并计算 SIM
- 监控训练目录并自动评测新 checkpoint

指标计算仍然依赖已经生成的 wav，因此普通流程必须先推理，再运行 WER 和 SIM。

### 1.3 MUSA 和 torchada

当前机器已验证的版本为：

```text
torch       2.7.1
torch_musa  2.7.1
torchada    0.1.59
torchaudio  2.7.1a0+95c61b4
```

`torch_musa` 提供 MUSA 设备支持。`torchada` 用于兼容部分原本按照 CUDA
API 编写的 PyTorch 代码。

这些版本是当前环境记录，不表示其他机器必须逐字相同；但 `torch`、
`torch_musa` 和 `torchaudio` 必须彼此兼容。

CosyVoice 批量推理脚本会尝试在加载 CosyVoice 前导入 `torchada`。如果没有安装，
导入失败会被忽略；也可以用 `--disable-torchada` 显式关闭。WER 和 SIM 中尽量
直接使用 `musa:0` 与 `.to(device)`，减少对 CUDA API 模拟的依赖。

### 1.4 当前脚本速查

| 脚本 | 作用 |
|---|---|
| `infer_cosyvoice_seedtts.py` | 单进程或单 shard 的 CosyVoice 推理 |
| `infer_cosyvoice.sh` | 支持 MUSA/CUDA 的多卡推理包装脚本 |
| `cal_wer.sh` | 将已有 wav 分片后计算 WER |
| `musa_cal_wer.sh` | 设置 MUSA 后端和当前英文路径的 WER 包装脚本 |
| `cal_sim.sh` | 将已有 wav 分片后计算 SIM |
| `musa_cal_sim.sh` | 设置 MUSA 后端、当前英文路径和 WavLM 路径的 SIM 包装脚本 |
| `cuda_cal_wer.sh` | 设置 CUDA 后端和当前英文路径的 WER 包装脚本 |
| `cuda_cal_sim.sh` | 设置 CUDA 后端、当前英文路径和 WavLM 路径的 SIM 包装脚本 |
| `eval_cosyvoice_checkpoint.sh` | 评测一个 LLM 或 flow checkpoint，输出英文 WER |
| `watch_cosyvoice_checkpoints.sh` | 持续发现并顺序评测训练 checkpoint |
| `prepare_ckpt.py` | 在 WER worker 启动前检查 ASR 模型能否加载 |
| `prepare_wavlm.py` | 解析 s3prl 源码并预加载 WavLM |

其中 `musa_cal_wer.sh`、`musa_cal_sim.sh`、`cuda_cal_wer.sh` 和
`cuda_cal_sim.sh` 都是便捷包装脚本；核心逻辑在 `infer_cosyvoice.sh`、
`cal_wer.sh` 和 `cal_sim.sh` 中，通过 `EVAL_BACKEND=musa|cuda` 选择后端。
迁移到其他机器时，优先检查这些包装脚本中的默认路径和环境变量。

## 2. 准备测试集

英文测试集：

```text
/home/cosyvoice-test/data/seedtts_testset/en/meta.lst
```

当前英文 `meta.lst` 有 1088 条记录。每行格式为：

```text
utt|prompt_text|prompt_wav|infer_text
```

部分其他列表可能有第五列参考语音：

```text
utt|prompt_text|prompt_wav|infer_text|infer_wav
```

例如：

```text
common_voice_en_xxx|This is prompt text.|prompt-wavs/xxx.wav|Text to synthesize.
```

其中第一列 `utt` 决定输出文件名。生成结果必须保存为：

```text
<output_dir>/<utt>.wav
```

否则后续 `cal_wer.sh` 和 `cal_sim.sh` 找不到该样本。

## 3. 准备 CosyVoice3 微调模型

CosyVoice3 加载模型时需要一个完整模型目录，其中不仅有微调后的
`llm.pt`，还需要：

- `cosyvoice3.yaml`
- `flow.pt`
- `hift.pt`
- `CosyVoice-BlankEN`
- `campplus.onnx`
- `speech_tokenizer_v3.onnx`
- 其他模型资源

如果只微调了 LLM，可以创建一个模型包装目录：

```text
/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test
```

该目录中的固定资源软链接到基础模型，只有 `llm.pt` 指向微调 checkpoint。

示意：

```bash
BASE=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B
TEST=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test
CKPT=/path/to/finetuned/llm.pt

mkdir -p "$TEST"

for asset in "$BASE"/*; do
  name=$(basename "$asset")
  if [ "$name" != "llm.pt" ]; then
    ln -sfn "$asset" "$TEST/$name"
  fi
done

ln -sfn "$CKPT" "$TEST/llm.pt"
```

检查实际使用的 checkpoint：

```bash
readlink -f \
  /home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test/llm.pt
```

当前 Python 推理脚本的模型目录优先级为：

```text
--model-dir
    > COSYVOICE_MODEL_DIR
    > /home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B
```

`infer_cosyvoice_seedtts.py` 本身的兜底默认值仍是基础模型
`Fun-CosyVoice3-0.5B`。但是当前 `infer_cosyvoice.sh` 在
`EVAL_BACKEND=musa` 时会默认设置：

```text
COSYVOICE_MODEL_DIR=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test
```

运行微调模型时，推荐显式确认当前模型目录：

```bash
export COSYVOICE_MODEL_DIR=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test
```

或者使用：

```bash
--model-dir /home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test
```

训练中自动评测不要求提前手工创建包装目录。
`eval_cosyvoice_checkpoint.sh` 会在每个 epoch 的输出目录下创建 `model/`，将基础
模型资源软链接进去，再把待评测 checkpoint 链接为 `llm.pt` 或 `flow.pt`。

## 4. 安装评测依赖

基础依赖：

```bash
cd /home/cosyvoice-test/seed-tts-eval
python3 -m pip install -r requirements.txt
```

主要依赖包括：

```text
funasr
zhconv
modelscope
librosa
jiwer>=3.0,<5
zhon
transformers
soundfile
scipy
tqdm
```

注意：

- `torch` 和 `torchaudio` 不要随意从 PyPI 覆盖。
- 应使用与当前 `torch_musa` 匹配的 PyTorch 和摩尔线程版 torchaudio。
- 英文 WER 不需要加载 FunASR；中文 WER 才使用 Paraformer。

可以检查关键版本：

```bash
python3 - <<'PY'
import torch
import torch_musa
import torchada
import torchaudio

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("has torch.musa:", hasattr(torch, "musa"))
print("MUSA available:", torch.musa.is_available())
PY
```

## 5. 使用 CosyVoice3 生成 Seed-TTS 音频

### 5.1 Python 推理脚本

脚本：

```text
/home/cosyvoice-test/seed-tts-eval/infer_cosyvoice_seedtts.py
```

它完成以下工作：

1. 读取 `meta.lst`。
2. 解析相对路径形式的 `prompt_wav`。
3. 给 CosyVoice3 的 `prompt_text` 添加：

   ```text
   You are a helpful assistant.<|endofprompt|>
   ```

4. 调用：

   ```python
   cosyvoice.inference_zero_shot(
       infer_text,
       prompt_text,
       prompt_wav,
       stream=False,
   )
   ```

5. 如果 CosyVoice 因文本切分返回多段音频，将它们拼接成一个 wav。
6. 保存为 `<utt>.wav`。
7. 支持多进程分片和断点续跑。

CosyVoice3 必须包含 `<|endofprompt|>`。如果没有这个标记，可能报错：

```text
<|endofprompt|> not detected in CosyVoice3 text or prompt_text
```

### 5.2 单卡小规模测试

建议先测试 5 条：

```bash
MUSA_VISIBLE_DEVICES=0 \
python3 /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice_seedtts.py \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  --device-backend musa \
  --limit 5
```

指定模型目录：

```bash
MUSA_VISIBLE_DEVICES=0 \
python3 /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice_seedtts.py \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  --model-dir /home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test \
  --device-backend musa \
  --limit 5
```

`--limit 5` 表示从筛选范围中总共取 5 条，不是每个 shard 各取 5 条。
`--start N` 可以跳过前 N 条，适合手工定位问题样本。

### 5.3 八卡全量推理

启动脚本：

```text
/home/cosyvoice-test/seed-tts-eval/infer_cosyvoice.sh
```

执行：

```bash
EVAL_BACKEND=musa \
COSYVOICE_MODEL_DIR=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test \
bash /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice.sh
```

如果不设置 `COSYVOICE_MODEL_DIR`，该脚本在 MUSA 后端默认使用
`Fun-CosyVoice3-0.5B-test`，在 CUDA 后端默认使用 `Fun-CosyVoice3-0.5B`。
`COSYVOICE_ROOT` 可以覆盖 CosyVoice 仓库路径。

当前包装脚本默认使用：

```text
meta.lst：/home/cosyvoice-test/data/seedtts_testset/en/meta.lst
输出目录：/home/cosyvoice-test/outputs/seedtts_eval/en
MUSA 设备：MUSA_DEVICE_LIST 指定；未指定时默认 8 个 worker，使用物理卡 0 到 7
CUDA 设备：CUDA_DEVICE_LIST 指定；未指定时默认 CUDA 0
```

需要使用其他测试集或输出目录时，可以直接把它们作为前两个参数传给
`infer_cosyvoice.sh`，也可以设置 `SEED_TTS_META` 和 `SEED_TTS_OUTPUT`：

```bash
EVAL_BACKEND=musa MUSA_DEVICE_LIST=4,5,6,7 \
bash /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice.sh \
  /path/to/meta.lst \
  /path/to/output_dir
```

脚本会按设备列表启动多个独立进程，例如：

```text
进程 0 -> MUSA 0 -> shard 0
进程 1 -> MUSA 1 -> shard 1
...
进程 7 -> MUSA 7 -> shard 7
```

这属于数据并行推理：每张卡各加载一份完整模型，并处理不同样本。它不是把一个
模型拆到 8 张卡上的模型并行。

英文 1088 条数据在 8 个 shard 下，每张卡处理 136 条，不重复、不遗漏。

### 5.4 全量重算和断点续跑

`infer_cosyvoice.sh` 当前带有：

```bash
--overwrite
```

因此每次运行都会覆盖已有 wav，适合更换 checkpoint 后重新生成全部结果。

如果只是继续之前中断的任务，应删除 `infer_cosyvoice.sh` 中的：

```bash
--overwrite
```

Python 脚本在没有 `--overwrite` 时会跳过已有 wav。

不希望修改包装脚本时，也可以直接按 5.2 节调用 Python 脚本且不传
`--overwrite`。

不要让不同 checkpoint 共用同一个输出目录并开启断点续跑。已有 wav 会被直接
跳过，最终目录可能混入多个 checkpoint 的结果。推荐每个模型使用独立目录：

```text
outputs/seedtts_eval/base/en
outputs/seedtts_eval/finetune_epoch_5/en
outputs/seedtts_eval/finetune_epoch_10/en
```

### 5.5 检查生成结果

英文全量应为 1088 个 wav：

```bash
find /home/cosyvoice-test/outputs/seedtts_eval/en \
  -maxdepth 1 -type f -name '*.wav' | wc -l
```

预期：

```text
1088
```

失败样本记录在：

```text
infer_failures_shard_00.tsv
infer_failures_shard_01.tsv
...
infer_failures_shard_07.tsv
```

这些文件可能是旧运行留下的。重新评测前应结合文件更新时间和 wav 总数判断，
不能只看到文件存在就认为当前运行失败。

推理脚本会记录单条失败，但当前不会因为存在失败样本自动返回非零状态。因此，
即使 shell 命令成功结束，也必须检查数量。列出缺失的 `utt`：

```bash
META=/home/cosyvoice-test/data/seedtts_testset/en/meta.lst
OUT=/home/cosyvoice-test/outputs/seedtts_eval/en

comm -23 \
  <(cut -d'|' -f1 "$META" | sort) \
  <(find "$OUT" -maxdepth 1 -type f -name '*.wav' -printf '%f\n' |
    sed 's/[.]wav$//' | sort)
```

命令没有输出才表示 `meta.lst` 中的每个 `utt` 都有对应 wav。

## 6. 计算英文 WER

### 6.1 WER 是什么

WER 是 Word Error Rate，即词错误率。英文流程使用
Whisper-large-v3 将生成语音转写为文本，再与 `meta.lst` 中的目标文本比较。

WER 越低越好。

当前仓库的 `average_wer.py` 先计算每条语音自己的 WER，再对所有语音做算术
平均：

```text
最终 WER = 所有单条 WER 的平均值
```

因此每条语音权重相同。这和把整个测试集的替换、删除、插入总数除以总词数得到的
corpus WER 不完全相同。比较不同实验时必须始终使用同一版本的
`run_wer.py`、`average_wer.py`、Whisper 模型和测试集。

### 6.2 准备本地 Whisper 模型

当前模型目录：

```text
/home/cosyvoice-test/data/models/whisper-large-v3
```

目录中至少应包含：

```text
config.json
generation_config.json
preprocessor_config.json
tokenizer_config.json
tokenizer.json
vocab.json
merges.txt
model.safetensors 或 pytorch_model.bin
```

默认本地模型路径由代码自动判断：

```text
/home/cosyvoice-test/data/models/whisper-large-v3
```

如果该目录存在，`prepare_ckpt.py` 和 `run_wer.py` 会优先使用它；否则回退到
`openai/whisper-large-v3`。也可以通过环境变量覆盖：

```bash
export WHISPER_MODEL=/path/to/whisper-large-v3
```

`prepare_ckpt.py` 只做模型预加载检查：

- `en` 只加载 Whisper
- `zh` 只加载 FunASR Paraformer
- 不传语言时加载两者

它不会生成 WER，也不会转换 checkpoint。

`prepare_ckpt.py` 的加载结果不会被多个 WER worker 共享。它的作用是提前暴露
模型文件缺失、格式错误或设备不可用等问题；随后每个 worker 仍会各加载一份
Whisper。

### 6.3 WER 的 MUSA 修改

关键修改：

- 使用 `device = "musa:0"`
- 每个 worker 通过 `MUSA_VISIBLE_DEVICES=$rank` 只看到一张物理卡
- worker 内始终使用逻辑设备 `musa:0`
- 英文和中文模型使用条件导入，英文评测不依赖 FunASR 初始化
- 任一 worker 失败时停止汇总，并保留 `/tmp` 中间结果
- 脚本使用绝对路径，可从任意目录执行
- 去掉原脚本中的 `sudo split`

### 6.4 八卡运行英文 WER

使用物理卡 0 到 7：

```bash
MUSA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
bash /home/cosyvoice-test/seed-tts-eval/cal_wer.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  en
```

参数依次是：

```text
参数 1：Seed-TTS meta.lst
参数 2：CosyVoice 生成 wav 的目录
参数 3：语言，英文为 en，中文为 zh
```

如果不设置 `MUSA_DEVICE_LIST`，`cal_wer.sh` 会根据 `NUM_GPUS`、
`ARNOLD_WORKER_GPU` 或默认值启动 worker。直接使用 `musa_cal_wer.sh` 时默认
`ARNOLD_WORKER_GPU=8`。

使用指定的非连续设备时，推荐只设置 `MUSA_DEVICE_LIST`，worker 数会自动等于
设备数量。例如使用物理卡 4、5、6、7：

```bash
MUSA_DEVICE_LIST=4,5,6,7 \
bash /home/cosyvoice-test/seed-tts-eval/cal_wer.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  en
```

如果同时设置 `NUM_GPUS` 或 `ARNOLD_WORKER_GPU`，它们的值必须与
`MUSA_DEVICE_LIST` 的设备数量一致。每个 worker 内部仍然使用逻辑设备
`musa:0`；`prepare_ckpt.py` 会自动使用设备列表中的第一张卡做预加载检查。

### 6.5 查看 WER

结果文件：

```text
/home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.wer
```

查看最终结果：

```bash
tail -1 \
  /home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.wer
```

输出示例：

```text
WER: 12.345%
```

文件前面的每一行还包含单条语音的参考文本、识别文本以及插入、删除、替换错误。

检查实际参与汇总的样本数：

```bash
grep -v -E '^(utt|WER:)' \
  /home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.wer |
  sed '/^$/d' | wc -l
```

全量英文测试预期为 1088。少于 1088 时，最终 WER 只是已有 wav 子集的结果，
不能直接与全量评测比较。

## 7. 计算说话人相似度 SIM

### 7.1 SIM 是什么

SIM 用于衡量生成语音与提示语音是否像同一个人。当前实现使用：

```text
WavLM-Large + ECAPA-TDNN speaker verification
```

对每条样本：

1. 从生成语音提取说话人 embedding。
2. 从 `prompt_wav` 提取说话人 embedding。
3. 计算两个 embedding 的余弦相似度。
4. 对所有样本求平均值。

SIM 越高通常表示说话人相似度越好。

单条余弦相似度理论范围是 `[-1, 1]`。当前仓库输出：

- `ASV`：所有有效样本相似度的算术平均
- `ASV-var`：所有有效样本相似度的总体方差，使用 NumPy `var`

本项目没有定义一个通用的“合格阈值”。SIM 受说话人模型、提示语音质量、语言、
音频长度和预处理影响，适合在完全相同的评测配置下比较不同模型，而不适合脱离
基线只看一个绝对数字。

### 7.2 为什么需要两个 WavLM 文件

SIM 需要两个不同文件，不能互相替代。

#### WavLM 基础模型

```text
/home/cosyvoice-test/data/models/wavlm_large.pt
```

约 1.2 GB，s3prl 格式，内部必须包含：

```python
{
    "cfg": ...,
    "model": ...
}
```

它用于构造和初始化 WavLM-Large 主干网络。

#### 说话人验证微调权重

```text
/home/cosyvoice-test/data/models/wavlm_large_finetune.pth
```

约 1.3 GB，内部包含：

```python
{
    "model": ...,
    "best_valid_eer": ...
}
```

它用于加载 WavLM + ECAPA 的说话人验证微调参数。

### 7.3 Hugging Face 目录为什么不能直接使用

以下目录：

```text
/home/cosyvoice-test/data/models/wavlm-large/
├── config.json
└── pytorch_model.bin
```

是 Hugging Face Transformers 格式。当前 s3prl SIM 实现不能直接将这个目录作为
`WAVLM_LARGE_CKPT`，因为它要求单文件中存在 `cfg` 和 `model`。

应使用：

```text
/home/cosyvoice-test/data/models/wavlm_large.pt
```

可以这样检查格式：

```bash
python3 - <<'PY'
import torch

p = "/home/cosyvoice-test/data/models/wavlm_large.pt"
checkpoint = torch.load(p, map_location="cpu", weights_only=False)
print(checkpoint.keys())
PY
```

正确输出应包含：

```text
cfg
model
```

### 7.4 SIM 的 MUSA 修改

原始代码使用：

```python
model.cuda(device)
wav.cuda(device)
```

已经改为：

```python
model.to(device)
wav.to(device)
```

启动 worker 时使用：

```bash
MUSA_VISIBLE_DEVICES=$rank
--device musa:0
```

其他修改：

- 结果文件后缀由错误的 `.wer` 改为 `.sim`
- 支持 8 卡 worker 分片
- 去掉 `sudo split`
- worker 进程异常时停止汇总
- 单条异常音频会跳过并写入 `.failures.tsv`，其余有效样本继续汇总
- 没有任何有效分数时主动报错
- `fire` 改为命令行单独运行时才导入，批量 SIM 不再依赖它
- 支持本地 `wavlm_large.pt`

### 7.5 为什么要先运行 prepare_wavlm.py

原始代码在每个 worker 中执行：

```python
torch.hub.load("s3prl/s3prl", "wavlm_large")
```

如果本地没有缓存，8 个 worker 会同时请求同一个模型。一个进程持有下载锁，其余
7 个进程等待，看起来就像程序卡住。

典型现象：

```text
/root/.cache/s3prl/download/...wavlm_large.pt.lock
```

同时 8 个 `thread-0*.sim.out` 都是 0 字节。

现在 `cal_sim.sh` 会先单进程运行：

```text
prepare_wavlm.py
```

确认基础模型可用后，再启动 `ARNOLD_WORKER_GPU` 指定数量的 MUSA worker。

#### s3prl 代码是什么

这里很容易把三个不同的东西混在一起：

| 内容 | 示例路径 | 用途 |
|---|---|---|
| s3prl 代码仓库 | `/root/.cache/torch/hub/s3prl_s3prl_main` | 提供 `hubconf.py`、WavLM 网络定义和加载代码 |
| WavLM 基础权重 | `/home/cosyvoice-test/data/models/wavlm_large.pt` | 初始化 WavLM-Large 主干 |
| 说话人验证微调权重 | `/home/cosyvoice-test/data/models/wavlm_large_finetune.pth` | 初始化 WavLM + ECAPA 说话人验证模型 |

`s3prl` 是一个语音表征学习工具库。当前项目不是只调用一个已经安装好的
Python 包，而是通过 PyTorch Torch Hub 读取 s3prl 仓库根目录中的
`hubconf.py`：

```python
torch.hub.load(s3prl_repo, "wavlm_local", ckpt=..., source="local")
```

因此，一个可用的本地 s3prl 目录至少应该类似：

```text
s3prl/
├── hubconf.py
├── s3prl/
│   ├── hub/
│   └── upstream/
└── ...
```

只有 `wavlm_large.pt` 不够，因为权重文件只保存参数，不包含完整的 Python
网络定义。只有 `pip install s3prl` 也不一定够，因为这里需要的是能作为
Torch Hub 仓库读取、并且根目录含有 `hubconf.py` 的源码目录。

#### 脚本会不会自动处理 s3prl

会。现在 `prepare_wavlm.py` 按以下顺序查找 s3prl：

1. `--s3prl-repo` 参数或 `S3PRL_HUB_DIR` 环境变量指定的目录。
2. 项目固定目录 `/home/cosyvoice-test/data/models/s3prl`。
3. 项目兼容目录 `/home/cosyvoice-test/data/models/s3prl_s3prl_main`。
4. 当前用户的 Torch Hub 缓存，例如
   `/root/.cache/torch/hub/s3prl_s3prl_main`。
5. 前面都没有找到时，通过 `torch.hub.load("s3prl/s3prl", ...)`
   从 GitHub 自动下载源码并写入 Torch Hub 缓存。

查找成功后，`cal_sim.sh` 会自动把实际目录导出为：

```bash
S3PRL_HUB_DIR=<实际找到的目录>
```

然后再启动所有 SIM worker。worker 只读取这个本地目录，不会各自重复下载
s3prl。

如果显式设置了 `S3PRL_HUB_DIR`，脚本会严格使用该目录。目录不存在或缺少
`hubconf.py` 时会直接报错，不会悄悄改用另一个版本。这可以避免评测过程中
意外切换 s3prl 代码版本。

#### 联网环境

如果机器可以访问 GitHub，不需要手动设置 `S3PRL_HUB_DIR`。首次运行时如果
本地没有 s3prl，脚本会自动下载，以后直接复用 Torch Hub 缓存：

```bash
bash /home/cosyvoice-test/seed-tts-eval/musa_cal_sim.sh
```

首次下载阶段可能会看到：

```text
No local s3prl repository was found. Downloading s3prl with Torch Hub...
```

下载完成后会打印最终使用的目录：

```text
WavLM-Large preparation completed; s3prl repo: /root/.cache/torch/hub/s3prl_s3prl_main
```

#### 离线或无法访问 GitHub

如果当前机器不能访问 GitHub，自动下载会失败。推荐在能联网的机器下载完整
s3prl 仓库，然后复制到项目的固定目录：

```bash
git clone https://github.com/s3prl/s3prl.git \
  /home/cosyvoice-test/data/models/s3prl
```

如果当前机器的 Torch Hub 缓存中已经有完整仓库，也可以复制过去：

```bash
cp -a /root/.cache/torch/hub/s3prl_s3prl_main \
  /home/cosyvoice-test/data/models/s3prl
```

复制后检查：

```bash
test -f /home/cosyvoice-test/data/models/s3prl/hubconf.py \
  && echo "s3prl repository is ready"
```

完全离线运行时可以设置：

```bash
export S3PRL_OFFLINE=1
```

此时若本地找不到 s3prl，脚本会立即给出明确错误，不会尝试访问 GitHub。

完全离线还必须设置本地 `WAVLM_LARGE_CKPT`。`S3PRL_OFFLINE=1` 只禁止下载
s3prl 源码；如果没有提供 `wavlm_large.pt`，本地 s3prl 的 `wavlm_large`
入口仍可能尝试下载基础权重。

也可以把源码放在任意目录，并手动指定：

```bash
export S3PRL_HUB_DIR=/your/local/path/s3prl
```

推荐优先放在 `/home/cosyvoice-test/data/models/s3prl`。项目目录比
`/root/.cache` 更便于迁移、备份和复现实验，也不容易因清理用户缓存而丢失。

#### 为什么之前仍然访问 GitHub

下面的调用虽然指定了本地 WavLM 权重：

```python
torch.hub.load("s3prl/s3prl", "wavlm_local", ckpt=checkpoint)
```

但第一个参数仍然是 GitHub 仓库名。PyTorch 必须先获得 s3prl 源码，才能解释
权重，因此仍会访问：

```text
https://github.com/s3prl/s3prl
```

这就是之前出现 `RemoteDisconnected` 的原因。现在找到本地仓库后使用：

```python
torch.hub.load(
    "/local/path/to/s3prl",
    "wavlm_local",
    ckpt=checkpoint,
    source="local",
)
```

`source="local"` 明确要求从本地读取源码，不再检查 GitHub。

### 7.6 八卡运行 SIM

包装脚本：

```text
/home/cosyvoice-test/seed-tts-eval/musa_cal_sim.sh
```

它已经设置：

```bash
export EVAL_BACKEND=musa
export S3PRL_HUB_DIR=/home/cosyvoice-test/s3prl
export S3PRL_OFFLINE=1
export WAVLM_LARGE_CKPT=/home/cosyvoice-test/data/models/wavlm_large.pt
export ARNOLD_WORKER_GPU=8
```

`WAVLM_LARGE_CKPT` 指向本地模型权重。`musa_cal_sim.sh` 默认按离线方式运行：
如果 `/home/cosyvoice-test/s3prl` 不存在或缺少 `hubconf.py`，会直接报错。
这种默认值适合固定评测机，能避免评测时意外访问 GitHub 或切换 s3prl 版本。

如果希望使用 7.5 节的自动查找或允许首次联网下载，不要用 `musa_cal_sim.sh`
的默认离线封装，改为直接调用 `cal_sim.sh`，并按需要设置或取消
`S3PRL_HUB_DIR`、`S3PRL_OFFLINE`。

直接运行：

```bash
bash /home/cosyvoice-test/seed-tts-eval/musa_cal_sim.sh
```

等价的完整命令：

```bash
ARNOLD_WORKER_GPU=8 \
WAVLM_LARGE_CKPT=/home/cosyvoice-test/data/models/wavlm_large.pt \
EVAL_BACKEND=musa \
bash /home/cosyvoice-test/seed-tts-eval/cal_sim.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  /home/cosyvoice-test/data/models/wavlm_large_finetune.pth
```

完全离线运行：

```bash
S3PRL_OFFLINE=1 \
ARNOLD_WORKER_GPU=8 \
WAVLM_LARGE_CKPT=/home/cosyvoice-test/data/models/wavlm_large.pt \
EVAL_BACKEND=musa \
bash /home/cosyvoice-test/seed-tts-eval/cal_sim.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  /home/cosyvoice-test/data/models/wavlm_large_finetune.pth
```

也可以把基础模型作为第四个参数：

```bash
EVAL_BACKEND=musa \
bash /home/cosyvoice-test/seed-tts-eval/cal_sim.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  /home/cosyvoice-test/data/models/wavlm_large_finetune.pth \
  /home/cosyvoice-test/data/models/wavlm_large.pt
```

当前 `cal_sim.sh` 和 WER 使用相同的设备选择规则。可以通过
`MUSA_DEVICE_LIST` 使用非连续设备，例如：

```bash
MUSA_DEVICE_LIST=4,5,6,7 \
bash /home/cosyvoice-test/seed-tts-eval/musa_cal_sim.sh
```

如果同时设置 `NUM_GPUS` 或 `ARNOLD_WORKER_GPU`，它们的值必须与设备列表数量
一致。每个 worker 内部使用逻辑设备 `musa:0`。如果训练正在占用部分卡，只应
填写确实有足够空闲显存的设备，或在独立机器上计算。

### 7.7 查看 SIM

结果文件：

```text
/home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.sim
```

查看最后两行：

```bash
tail -2 \
  /home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.sim
```

输出示例：

```text
ASV: 0.678
ASV-var: 0.012
```

其中：

- `ASV`：平均说话人相似度
- `ASV-var`：所有样本相似度的方差

检查实际得到相似度的样本数：

```bash
grep -v -E '^(ASV:|ASV-var:)' \
  /home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.sim |
  sed '/^$/d' | wc -l
```

理想的全量英文测试应有 1088 条相似度。`verification_pair_list_v2.py` 会跳过
不存在的生成音频或提示音频；异常音频会记录到 `.failures.tsv`。因此只查看
`ASV` 不足以证明全量评测完整。

当前工作区现有结果是：

```text
wav 数：1088
WER 有效样本：1088
SIM 有效样本：1085
SIM failures：3
```

这 3 条生成音频长度只有 0.04 到 0.08 秒，低于当前 WavLM 检查要求的 0.1 秒。
所以当前 `ASV` 实际是 1085 条有效样本的平均值。汇报结果时应同时写出
`ASV`、有效样本数和失败样本数。

## 8. 常见提示和错误

### 8.1 ESPnet is not installed

提示：

```text
ESPnet is not installed, cannot use espnet_hubert upstream
```

这不是当前任务的错误。

s3prl 支持很多不同 upstream，其中包括 `espnet_hubert`。当前使用的是
`wavlm_large`，不需要 ESPnet，可以忽略。

### 8.2 SIM 一直没有第一条结果

当前 `cal_sim.sh` 会先运行单进程 `prepare_wavlm.py`，准备阶段完成前不会启动
正式 SIM worker。因此刚开始没有分数文件增长，可能只是正在加载约 1.2 GB 的
WavLM 基础权重。

检查：

```bash
find /root/.cache/s3prl/download -maxdepth 1 -type f -ls
```

如果只有：

```text
*.wavlm_large.pt.lock
```

说明旧运行或缺少本地基础权重时正在下载模型。当前实现不应再让 8 个 worker
同时争抢该锁。

解决：

1. 按 `Ctrl+C` 停止旧运行。
2. 准备本地 s3prl 格式 `wavlm_large.pt`。
3. 设置 `WAVLM_LARGE_CKPT`。
4. 重新运行 `musa_cal_sim.sh`。

还可以确认当前究竟停在准备阶段还是评分阶段：

```bash
ps -ef | grep -E '[p]repare_wavlm|[v]erification_pair_list_v2'
```

### 8.3 MUSA driver initialization failed

提示：

```text
MUSA driver initialization failed
```

说明当前进程无法访问 MUSA 驱动或 MUSA 设备。检查：

```bash
python3 - <<'PY'
import torch
print(hasattr(torch, "musa"))
print(torch.musa.is_available())
print(torch.musa.device_count())
PY
```

如果 `is_available()` 为 `False`，需要先解决容器设备挂载、驱动或运行环境问题。

### 8.4 WER 缺少 jiwer 或 zhon

报错：

```text
ModuleNotFoundError: No module named 'jiwer'
```

或：

```text
ModuleNotFoundError: No module named 'zhon'
```

安装：

```bash
pip install "jiwer>=3.0,<5" zhon
```

当前代码使用新版 `jiwer` 接口：

```python
from jiwer import process_words
```

### 8.5 输出 wav 数量不足

检查：

```bash
wc -l /home/cosyvoice-test/data/seedtts_testset/en/meta.lst

find /home/cosyvoice-test/outputs/seedtts_eval/en \
  -maxdepth 1 -type f -name '*.wav' | wc -l
```

两个数字应一致。英文当前都是 1088。

如果不一致，查看：

```bash
cat /home/cosyvoice-test/outputs/seedtts_eval/en/infer_failures_shard_*.tsv
```

然后使用不带 `--overwrite` 的 Python 推理命令补齐缺失样本；如果使用
`infer_cosyvoice.sh`，需要先移除其中固定的 `--overwrite`。

### 8.6 WER 或 SIM worker 失败

当前脚本会输出类似：

```text
WER worker failed; intermediate files kept in /tmp/thread_metas_xxx/results
```

或：

```text
SIM worker failed; intermediate files kept in /tmp/thread_metas_xxx/results
```

中间文件不会立即删除，可以检查：

```bash
ls -lh /tmp/thread_metas_xxx/results
```

SIM 的单条失败详情最终合并保存在：

```text
<output_dir>/wav_res_ref_text.sim.failures.tsv
```

极短音频可能在 WavLM 下采样后只剩一个时间步，无法计算可靠的说话人
embedding。当前实现会跳过短于 0.1 秒的音频，并在最终结果旁记录原因。

SIM 使用单样本串行推理。每条样本之间还包含 CPU 读取音频、重采样和结果写入，
所以显存会持续占用，但 GPU 利用率通常呈脉冲状，不一定长期保持高占用。

### 8.7 自动评测没有发现 checkpoint

`watch_cosyvoice_checkpoints.sh` 只扫描指定目录的第一层普通文件：

```bash
find "$MODEL_DIR" -maxdepth 1 -type f -name 'epoch_*_whole.pt'
```

每个 checkpoint 还必须有同名 YAML：

```text
epoch_0_whole.pt
epoch_0_whole.yaml
```

检查 watcher 实际能否看到文件：

```bash
MODEL_DIR=/path/to/the/real/checkpoint/directory

find "$MODEL_DIR" -maxdepth 1 -type f -name 'epoch_*_whole.pt' -print
ls -l "$MODEL_DIR"/epoch_0_whole.{pt,yaml}
```

常见原因包括：

- 传入的是 checkpoint 目录的上一级，真实文件在子目录中
- checkpoint 是软链接，`find -type f` 不会把链接本身当普通文件
- 训练尚未写出对应 YAML
- 文件名不是严格的 `epoch_<数字>_whole.pt`

需要观察详细判断过程时：

```bash
POLL_SECONDS=5 bash -x watch_cosyvoice_checkpoints.sh ... \
  2>&1 | tee watch_eval.log
```

停止 watcher：

```bash
pkill -f watch_cosyvoice_checkpoints.sh
```

### 8.8 自动评测完成但样本数不足

`eval_cosyvoice_checkpoint.sh` 当前会对已经存在的 wav 计算 WER。只要至少存在
一个有效 wav，流程就可能生成 `wer.txt` 和 `.complete`。因此 `.complete`
表示该次脚本流程走完，不保证目标样本全部生成。

自动评测完成后至少检查：

```bash
EPOCH_OUT=/path/to/checkpoint_eval/epoch_1

find "$EPOCH_OUT/wavs" -maxdepth 1 -type f -name '*.wav' | wc -l
grep -v -E '^(utt|WER:)' "$EPOCH_OUT/wer.txt" | sed '/^$/d' | wc -l
find "$EPOCH_OUT/wavs" -maxdepth 1 -name 'infer_failures_shard_*.tsv' -size +0c -print
```

如果是全量英文评测，前两个数字都应为 1088。数量不足时应先处理失败样本，再删除
该 epoch 的 `.complete` 后续跑。

## 9. 训练中自动评测 checkpoint

这一节属于进阶功能。只想评估一个确定 checkpoint 时，可以跳过。

### 9.1 评测单个 checkpoint

脚本：

```text
eval_cosyvoice_checkpoint.sh
```

它会：

1. 创建临时完整模型目录。
2. 将基础模型固定资源软链接进去。
3. 将指定 checkpoint 链接为 `llm.pt` 或 `flow.pt`。
4. 使用 CosyVoice 生成 Seed-TTS 音频。
5. 计算英文 WER。

调用：

```bash
EVAL_BACKEND=musa \
bash /home/cosyvoice-test/seed-tts-eval/eval_cosyvoice_checkpoint.sh \
  /path/to/epoch_4_whole.pt \
  /home/cosyvoice-test/outputs/checkpoint_eval/epoch_5 \
  0
```

参数：

```text
参数 1：checkpoint
参数 2：本次评测输出目录
参数 3：设备编号或逗号分隔列表，默认 0
参数 4：只评测前 N 条，可省略
参数 5：llm 或 flow，默认 llm
```

例如只测试前 20 条：

```bash
bash /home/cosyvoice-test/seed-tts-eval/eval_cosyvoice_checkpoint.sh \
  /path/to/epoch_4_whole.pt \
  /home/cosyvoice-test/outputs/checkpoint_eval/smoke_test \
  0 \
  20 \
  llm
```

多卡评测一个 flow checkpoint：

```bash
bash /home/cosyvoice-test/seed-tts-eval/eval_cosyvoice_checkpoint.sh \
  /path/to/epoch_0_whole.pt \
  /home/cosyvoice-test/outputs/checkpoint_eval/flow_epoch_1 \
  4,5,6,7 \
  "" \
  flow
```

第四个参数传空字符串表示不限制样本数。每张卡启动一个 CosyVoice 推理 shard；
推理结束后 Whisper WER 也按相同设备列表分片。后端由 `EVAL_BACKEND`、
`MUSA_DEVICE_LIST`、`CUDA_DEVICE_LIST` 或 `DEFAULT_EVAL_BACKEND` 决定；MUSA
环境中推荐显式设置 `EVAL_BACKEND=musa`。

输出目录结构：

```text
flow_epoch_1/
├── model/                 # 基础资源与待评测 checkpoint 的软链接
├── wavs/                  # 生成音频和每个 shard 的失败记录
├── wav_res_ref_text       # 实际找到的 wav 与参考文本
├── wer.txt                # 单条结果和最终 WER
└── .complete              # 脚本流程完成标记
```

默认基础模型、测试集和 CosyVoice 仓库可以分别用以下环境变量覆盖：

```bash
BASE_MODEL_DIR=/path/to/Fun-CosyVoice3-0.5B
SEED_TTS_META=/path/to/meta.lst
COSYVOICE_ROOT=/path/to/CosyVoice
```

在 MUSA 后端中，如果不设置 `BASE_MODEL_DIR`，默认使用：

```text
/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test
```

CUDA 后端默认使用：

```text
/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B
```

评测默认跳过输出目录中已经存在的 wav，适合在中断后续跑。设置
`EVAL_OVERWRITE=1` 才会覆盖全部已有 wav。

同一个输出目录只能对应同一个 checkpoint、component、测试集和推理配置。如果
更换了任意一项，应该换新目录；否则旧 wav 可能被跳过并混入新结果。

### 9.2 监控训练目录

脚本：

```text
watch_cosyvoice_checkpoints.sh
```

它会监控训练目录中的：

```text
epoch_*_whole.pt
```

并按照设定间隔自动评测 WER。

checkpoint 发现规则是：

1. 只扫描 `MODEL_DIR` 第一层，不递归子目录。
2. 只接受普通文件 `epoch_<数字>_whole.pt`。
3. 必须同时存在 `epoch_<数字>_whole.yaml`，用于判断 checkpoint 已完成保存。
4. 文件编号从 0 开始，`epoch_0_whole.pt` 表示第 1 个完成的 epoch。
5. 多个待评测 checkpoint 会按版本号顺序逐个评测，不是只评最新一个。

示例：

```bash
bash /home/cosyvoice-test/seed-tts-eval/watch_cosyvoice_checkpoints.sh \
  /path/to/training/checkpoints \
  /home/cosyvoice-test/outputs/checkpoint_history \
  5 \
  0 \
  llm
```

表示：

- 每 5 个完成的 epoch 评测一次
- 使用 MUSA 0
- 替换 LLM checkpoint

例如 interval 为 5 时会评测：

```text
epoch_4_whole.pt  -> 完成第 5 个 epoch
epoch_9_whole.pt  -> 完成第 10 个 epoch
epoch_14_whole.pt -> 完成第 15 个 epoch
```

`DEVICE` 也可以传入逗号分隔的多卡列表。例如使用 8 张卡并行完成 CosyVoice
推理和 Whisper WER：

```bash
WER_WINDOW=5 WER_DELTA=0.1 \
bash /home/cosyvoice-test/seed-tts-eval/watch_cosyvoice_checkpoints.sh \
  /path/to/training/checkpoints \
  /home/cosyvoice-test/outputs/checkpoint_history \
  1 \
  0,1,2,3,4,5,6,7 \
  flow
```

每张卡负责一个 Seed-TTS 数据分片，全部推理完成后，WER 也会按相同设备列表
并行计算。若训练任务正在占用部分卡，只应填写确实有足够空闲显存的设备，例如：

```text
4,5,6,7
```

自动评测默认支持断点续跑。评测被停止后，重新启动 watcher 会跳过已经生成的
wav，只补齐剩余样本。需要强制重新生成全部音频时设置：

```bash
EVAL_OVERWRITE=1
```

如果某个 epoch 已经存在 `.complete`，watcher 不会再次调用评测脚本。重新评测
时至少要删除：

```bash
rm /path/to/output_root/epoch_1/.complete
```

如果 checkpoint、模型组件、测试集或推理参数发生了变化，推荐直接删除整个
`epoch_1` 输出目录或使用新的 `OUTPUT_ROOT`，不要只删除 `.complete` 后复用旧
wav。

历史结果写入：

```text
wer_history.tsv
```

每行记录：

```text
完成的 epoch 数<TAB>WER 数值<TAB>wer.txt 路径
```

同一个 `OUTPUT_ROOT` 中，相同 epoch 只记录一次。因此不同训练实验、不同模型
组件或不同测试集必须使用独立的输出根目录。

常用环境变量：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `POLL_SECONDS` | `30` | 没有待评测 checkpoint 时的轮询间隔 |
| `EVAL_LIMIT` | 空 | 只评测前 N 条，用于冒烟测试 |
| `EVAL_OVERWRITE` | `0` | 设为 `1` 时覆盖已有 wav |
| `WER_WINDOW` | `3` | 早停判断使用最近多少次 WER |
| `WER_DELTA` | `0.1` | 最近窗口中最大 WER 与最小 WER 的允许差值，单位是百分点 |
| `EARLY_STOP_FILE` | `MODEL_DIR/STOP_TRAIN` | 收敛时写入的停止请求文件 |
| `MAX_EPOCH` | 空 | 对应 `epoch_N/.complete` 出现后退出 watcher |

`EVAL_LIMIT` 产生的是子集 WER。不要把冒烟测试结果和全量 WER 放在同一个
`OUTPUT_ROOT`，否则 `.complete` 和 `wer_history.tsv` 会让 watcher 认为该 epoch
已经完成正式评测。

`EVAL_LIMIT=N` 会同时限制 CosyVoice 推理和 WER 清单。即使 `wavs/` 中残留更多
旧文件，本次 WER 也只使用 `meta.lst` 的前 N 条。日志中的第一个 `N/N` 是扫描
评分清单，随后 Whisper 的进度条才是实际参与 WER 的 wav 数。

早停规则不是“WER 连续下降不明显”，而是：

```text
最近 WER_WINDOW 次 WER 的最大值 - 最小值 <= WER_DELTA
```

这表示 WER 进入一个窄区间，即使它停在较差水平也可能触发。触发后 watcher 只会
写出 `STOP_TRAIN` 并退出；当前 CosyVoice 训练程序不会自动读取该文件，必须由
训练启动脚本或外部调度器主动检查，才能真正停止训练。

建议保存 watcher 日志：

```bash
EVAL_BACKEND=musa \
POLL_SECONDS=5 \
WER_WINDOW=5 \
WER_DELTA=0.1 \
bash /home/cosyvoice-test/seed-tts-eval/watch_cosyvoice_checkpoints.sh \
  /path/to/the/real/checkpoint/directory \
  /home/cosyvoice-test/outputs/flow_seedtts_eval \
  1 \
  4,5,6,7 \
  flow 2>&1 | tee watch_flow.log
```

评测设备必须有足够显存，并且不能和训练进程争抢同一组卡。自动评测当前只计算
英文 WER；需要 SIM 时，应在对应 epoch 推理完成后另外运行 `cal_sim.sh`。

## 10. 推荐的完整执行顺序

### 第一步：确认模型和测试集

```bash
MODEL_DIR=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test

test -f "$MODEL_DIR/cosyvoice3.yaml"
readlink -f "$MODEL_DIR/llm.pt"
readlink -f "$MODEL_DIR/flow.pt"

wc -l /home/cosyvoice-test/data/seedtts_testset/en/meta.lst
```

### 第二步：小规模生成测试

```bash
MUSA_VISIBLE_DEVICES=0 \
python3 /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice_seedtts.py \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/smoke_test \
  --model-dir /home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test \
  --device-backend musa \
  --limit 5 \
  --overwrite
```

### 第三步：八卡生成全部音频

```bash
EVAL_BACKEND=musa \
MUSA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
COSYVOICE_MODEL_DIR=/home/cosyvoice-test/pretrained_models/Fun-CosyVoice3-0.5B-test \
bash /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice.sh
```

### 第四步：确认数量

```bash
find /home/cosyvoice-test/outputs/seedtts_eval/en \
  -maxdepth 1 -type f -name '*.wav' | wc -l
```

预期为：

```text
1088
```

同时确认没有缺失 `utt`，方法见 5.5 节。不要仅凭推理命令退出码判断全量成功。

### 第五步：计算 WER

```bash
MUSA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
bash /home/cosyvoice-test/seed-tts-eval/cal_wer.sh \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  en
```

确认 WER 文件中也有 1088 条单句结果，方法见 6.5 节。

### 第六步：计算 SIM

```bash
bash /home/cosyvoice-test/seed-tts-eval/musa_cal_sim.sh
```

确认 SIM 文件中有 1088 条有效相似度，或检查
`wav_res_ref_text.sim.failures.tsv`，方法见 7.7 节。

### 第七步：查看最终指标

```bash
tail -1 \
  /home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.wer

tail -2 \
  /home/cosyvoice-test/outputs/seedtts_eval/en/wav_res_ref_text.sim
```

## 11. 本次适配修改总览

### CosyVoice 推理

- 新增 Seed-TTS `meta.lst` 批量推理脚本
- 支持 CosyVoice3 `<|endofprompt|>` 格式
- 支持 torchada/MUSA
- 支持 `COSYVOICE_MODEL_DIR` 和 `COSYVOICE_ROOT`
- 支持多段音频拼接
- 支持八卡数据分片
- 支持断点续跑和全量覆盖
- 多卡情况下 `--limit` 按总样本数生效
- 支持每个 shard 独立失败记录

### WER

- Whisper-large-v3 改为本地模型目录
- 支持 `EVAL_BACKEND=musa|cuda` 选择后端
- 英文和中文模型条件加载
- 八卡进程使用 `MUSA_VISIBLE_DEVICES`
- 支持 `MUSA_DEVICE_LIST` 选择非连续设备
- 去掉 `sudo`
- 使用绝对脚本路径
- worker 失败时停止错误汇总
- 修正 `average_wer.py` 路径

### SIM

- `.cuda()` 改为 `.to(device)`
- 支持 `musa:0` 和 `cuda:0`
- 多卡进程使用后端对应的可见设备变量
- 支持 `MUSA_DEVICE_LIST` 或 `CUDA_DEVICE_LIST` 选择设备
- 增加本地 s3prl `wavlm_large.pt` 支持
- 增加单进程 WavLM 预加载
- 避免八个 worker 争抢下载锁
- 结果后缀改为 `.sim`
- 增加单条失败记录
- worker 失败时停止错误汇总
- 将不必要的 `fire` 改为延迟导入

### 自动 checkpoint 评测

- 支持将单个 LLM 或 flow checkpoint 临时组装成完整模型目录
- 支持单卡或逗号分隔的多卡推理和 WER
- 支持监控 `epoch_*_whole.pt` 与同名 YAML
- 支持按 epoch 间隔顺序评测
- 支持 wav 断点续跑和 `.complete` 完成标记
- 支持 WER 历史记录和基于窗口波动范围的停止请求文件
- 当前不自动计算 SIM，也不直接终止 CosyVoice 训练进程

完成以上步骤后，就可以用统一的 Seed-TTS 测试集比较：

- CosyVoice3 基础模型
- 不同微调 checkpoint
- 不同训练 epoch
- 不同训练配置

比较时应保持测试集、Whisper 模型、WavLM 模型和推理参数一致，否则不同实验的
WER/SIM 不具备严格可比性。还应记录参与汇总的有效样本数，避免把缺失样本后的
子集结果误当作完整测试集结果。
