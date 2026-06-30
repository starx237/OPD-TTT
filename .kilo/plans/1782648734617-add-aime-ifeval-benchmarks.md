# 添加 AIME'24 和 IF-eval Benchmark 评测

## 目标

为 OPD-TTT 训练添加 AIME'24（数学推理）和 IF-eval（指令遵循）两个 benchmark，用于评估 SFT baseline 与 OPD checkpoint 的 data efficiency。

## 决策

1. **独立脚本方案**：复用 `test_generation.py` 的模型加载模式，不依赖 OpenCompass
2. **评测模型**：SFT baseline（stage2 SFT checkpoint）+ OPD checkpoint（step 150）
3. **IF-eval 评测代码**：从 GitHub 下载 Google Research 官方代码

## 文件结构

```
scripts/
  _model_loader.py          # 共享模型加载工具
  eval_aime.py              # AIME'24 评测脚本
  eval_ifeval.py            # IF-eval 评测脚本
  instruction_following_eval/  # 下载的官方 IF-eval 代码
    instructions_registry.py
    instructions_util.py
results/
  aime_results.json         # AIME 逐题结果
  ifeval_results.json       # IF-eval 逐题结果
  eval_summary.txt          # 汇总对比表
```

## 实现步骤

### 步骤1：创建共享模型加载工具 `_model_loader.py`

从 `test_generation.py` 提取模型加载逻辑为可复用函数：

```python
def load_opdttt_model(ckpt_path=None, device="cuda"):
    """
    加载 OPD-TTT 模型。
    - ckpt_path=None: 加载 SFT baseline（HF checkpoint）
    - ckpt_path="path/to/dcp": 先加载SFT HF，再用DCP覆盖权重
    返回 (model, tokenizer)
    """
    # 1. 加载 config（复用 test_generation.py 的 config 设置）
    # 2. OPDTTTForCausalLM.from_pretrained(SFT HF checkpoint, ...)
    # 3. 如果有 ckpt_path: ckpt_to_state_dict + load_state_dict
    # 4. model.cuda().eval()
    # 5. 加载 tokenizer
```

关键配置（与 `test_generation.py` 一致）：
- `config_path = "model_assets/llama_500m_config"`
- `tokenizer_path = "model_assets/tokenizer"`
- SFT HF checkpoint: `data/output/500m_stage2_sft/checkpoints/global_step_3000/hf_ckpt`
- OPD DCP checkpoint: `data/output/500m_stage2_opd/checkpoints/global_step_150/global_step_150`
- `config.opdttt_mode=True`, `opdttt_layers=[0,6,12,18]`, `ttt_lr=3`, `ttt_chunk=1024`, `ttt_proj=True`, `lambda_kl=0.5`, `lambda_ntp=1.0`, `lambda_lm=0.0`, `lambda_align_rep=0.0`
- `teacher_hidden_size` 从 `model_assets/teacher_qwen2.5_7b` 的 config 读取

### 步骤2：下载 AIME'24 数据集

AIME 2024 有 30 道数学竞赛题，答案为 0-999 的整数。

数据来源：HuggingFace `datasets.load_dataset("Maxwell-Jia/AIME_2024")` 或备选 `AI-MO/aimo-validation-aime`。

如果网络受限，手动下载并保存为 `data/aime_2024.jsonl`，格式：
```json
{"id": 1, "problem": "...", "answer": 42}
```

### 步骤3：实现 AIME'24 评测脚本 `eval_aime.py`

**Prompt 格式**（zero-shot）：
```
Solve the following math problem. Show your work step by step. At the end, write your final answer as "The answer is: [integer]".

Problem: {problem}
```

**生成参数**：
- `temperature=0`（greedy），`do_sample=False`
- `max_new_tokens=1024`
- `pad_token_id=tokenizer.eos_token_id`

**答案提取**（按优先级）：
1. 正则匹配 `The answer is:\s*(\d+)`
2. 正则匹配 `\\boxed\{(\d+)\}`
3. 正则匹配 `answer is\s*(\d+)`
4. 提取文本中最后一个整数

**评分**：
- 提取的整数 == ground truth → 正确
- Metric: accuracy = 正确数 / 总题数

**输出** `results/aime_results.json`：
```json
{
  "SFT": {"accuracy": 0.03, "correct": 1, "total": 30, "samples": [...]},
  "OPD-150": {"accuracy": 0.0, "correct": 0, "total": 30, "samples": [...]}
}
```

### 步骤4：下载 IF-eval 数据集和官方评测代码

**数据集**：HuggingFace `datasets.load_dataset("google/IFEval")` 或 `HuggingFaceH4/ifeval`（541 条带可验证指令的 prompt）。

**官方评测代码**：从 GitHub 下载 3 个文件到 `scripts/instruction_following_eval/`：
- `https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/instructions_registry.py`
- `https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/instructions_util.py`
- `https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/evaluation_main.py`

如果网络受限，用 `HF_ENDPOINT=https://hf-mirror.com` 或手动下载。

### 步骤5：实现 IF-eval 评测脚本 `eval_ifeval.py`

**Prompt 格式**：直接使用数据集中的 prompt（已包含指令约束）。

**生成参数**：
- `temperature=0`（greedy），`do_sample=False`
- `max_new_tokens=512`
- `pad_token_id=tokenizer.eos_token_id`

**评分**：调用官方 `evaluation_main.py` 的逻辑：
```python
from instructions_following_eval.instructions_registry import INSTRUCTION_DICT
from instructions_following_eval.instructions_util import InstructionsUtil

# 对每个 prompt 的每条指令，检查 response 是否满足
for instruction in prompt_instructions:
    instruction_id = instruction["instruction_id"]
    args = instruction["instruction_id_args"]
    checker = INSTRUCTION_DICT[instruction_id]
    followed = checker.check_following(args, response)  # True/False
```

**Metrics**：
- **Strict accuracy**: 所有指令都满足的比例
- **Loose accuracy**: 至少一条指令满足的比例（宽松模式）

**输出** `results/ifeval_results.json`：
```json
{
  "SFT": {"strict_accuracy": 0.15, "loose_accuracy": 0.30, "total": 541, "samples": [...]},
  "OPD-150": {"strict_accuracy": 0.05, "loose_accuracy": 0.12, "total": 541, "samples": [...]}
}
```

### 步骤6：运行评测

```bash
# AIME'24
CUDA_VISIBLE_DEVICES=0 python scripts/eval_aime.py --output results/aime_results.json

# IF-eval
CUDA_VISIBLE_DEVICES=0 python scripts/eval_ifeval.py --output results/ifeval_results.json
```

两个脚本都内置 SFT 和 OPD 两个模型的评测，通过 `_model_loader.py` 加载。

### 步骤7：更新 `documents/opd_report.md`

在 data efficiency 评估部分添加 AIME'24 和 IF-eval 结果表：

```markdown
#### Benchmark 评估

| 模型 | AIME'24 准确率 | IF-eval (strict) | IF-eval (loose) |
|------|---------------|------------------|-----------------|
| SFT | X% | X% | X% |
| OPD-150 | X% | X% | X% |
```

## 风险与注意事项

1. **网络访问**：HuggingFace 和 GitHub 可能需要镜像（`HF_ENDPOINT=https://hf-mirror.com`）。如果无法下载，需要手动准备数据文件。
2. **模型能力**：500M 模型在 AIME'24 上可能接近 0%（AIME 是高难度竞赛）。重点看 SFT vs OPD 的**相对差异**，而非绝对分数。
3. **模式崩溃**：OPD 模型有 71% 重复率，AIME/IF-eval 分数可能显著低于 SFT。这本身就是有价值的数据——证明模式崩溃对 benchmark 性能的影响。
4. **IF-eval 代码兼容性**：官方代码可能依赖特定版本的 NLTK 等 NLP 库（用于句子分割）。如果依赖缺失，需要 `pip install nltk` 或回退到简单的句号分割。
5. **DCP checkpoint 加载**：`ckpt_to_state_dict` 在单进程模式下可能打印 warning，可以忽略。

## 验证

1. AIME'24：检查答案提取是否正确（手动看几条 sample）
2. IF-eval：检查约束检查是否正常工作（手动验证几条 sample）
3. 两个模型的分数都应输出到 JSON 和汇总表
4. 如果 OPD 分数 < SFT 分数，与模式崩溃结论一致
