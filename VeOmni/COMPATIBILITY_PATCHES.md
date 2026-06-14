# VeOmni Compatibility Patches

本文档记录了 VeOmni 框架与特定环境的兼容性补丁，主要用于支持 PyTorch 2.5.1 和 transformers 4.57.3 环境。

## 环境信息

- **PyTorch**: 2.5.1 with CUDA 12.4
- **Transformers**: 4.57.3
- **Flash Attention**: 2.6.3
- **Python**: 3.11

## 已添加的兼容性补丁

### 1. 配置文件扁平结构支持 (`parser.py`)

**位置**: `veomni/arguments/parser.py`

**问题**: VeOmni 使用嵌套配置结构（如 `accelerator.fsdp_config.offload`），但许多训练脚本使用扁平结构（如 `enable_fsdf_offload`）。

**解决方案**: 添加了 `_map_flat_to_nested()` 函数，在配置解析时自动将扁平结构映射到嵌套结构。

**支持的扁平字段**:
- `enable_fsdf_offload` → `accelerator.fsdf_config.offload`
- `enable_activation_offload` → `accelerator.offload_config.enable_activation`
- `enable_gradient_checkpointing` → `gradient_checkpointing.enable`
- `enable_full_shard` → `accelerator.fsdf_config.fsdf_mode`
- `data_parallel_mode` → `accelerator.fsdf_config.fsdf_mode`
- `ulysses_parallel_size` → `accelerator.ulysses_size`
- `enable_mixed_precision` → `accelerator.fsdf_config.mixed_precision.enable`

### 2. PyTorch 2.5.x DCP API 兼容性 (`dcp_checkpointer.py`)

**位置**: `veomni/checkpoint/dcp_checkpointer.py`

**问题**: PyTorch 2.5.x 中 `torch.distributed.checkpoint.load()` 不再接受 `no_dist` 参数。

**解决方案**: 移除过时的 `no_dist=True` 参数：

```python
# 旧代码（PyTorch 2.4.x）
load(
    state_dict,
    checkpoint_id=checkpoint_path,
    storage_reader=FileSystemReader(checkpoint_path),
    no_dist=True,  # ← 在 PyTorch 2.5+ 中不再支持
)

# 新代码（PyTorch 2.5.x 兼容）
load(
    state_dict,
    checkpoint_id=checkpoint_path,
    storage_reader=FileSystemReader(checkpoint_path),
    # 不再传递 no_dist 参数
)
```

### 3. FSDP2 + Gradient Checkpointing DTensor 兼容性

**问题**: 在 FSDP2 下使用 gradient checkpointing 时，重计算阶段的输入是普通 Tensor 而参数是 DTensor，导致 `RuntimeError: aten.mul.Tensor: got mixed torch.Tensor and DTensor`。

**当前解决方案**: 暂时禁用 gradient checkpointing (`enable_gradient_checkpointing: false`)。

**长期解决方案**: 需要修改 gradient checkpointing 实现以正确处理 DTensor。

### 4. HuggingFace Backend 3-Tuple Loss 返回兼容性 (`ops/__init__.py`)

**位置**: `veomni/ops/__init__.py`

**问题**: 当 `MODELING_BACKEND=hf` 时，HuggingFace 原始模型期望 loss 函数返回标量，但 VeOmni 的 loss wrapper 返回 `(loss, logits, fused_linear_aux)` 元组。

**解决方案**: 添加最小化 monkey-patch 来处理元组返回值：

```python
def _patch_hf_model_for_veomni_loss():
    """Minimal patch for HuggingFace models to handle VeOmni's tuple loss return."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    original_forward = LlamaForCausalLM.forward

    def patched_forward(self, *args, **kwargs):
        outputs = original_forward(self, *args, **kwargs)

        # Check if loss is a tuple (VeOmni 3-tuple return)
        if hasattr(outputs, "loss") and isinstance(outputs.loss, tuple):
            # Unpack: (loss_tensor, logits, fused_linear_aux)
            loss_tensor, logits, fused_linear_aux = outputs.loss
            outputs.loss = loss_tensor
            # Update logits if it was modified
            if logits is not None:
                outputs.logits = logits

        return outputs

    LlamaForCausalLM.forward = patched_forward
```

### 5. FSDP2 logits.float() OOM 修复 (`cross_entropy/__init__.py`)

**位置**: `veomni/ops/kernels/cross_entropy/__init__.py`

**问题**: transformers 4.57.3 的 loss 计算强制将 logits 转换为 float32，对于 32k 序列长度会消耗 4.2GB 显存（bfloat16 为 2.1GB），导致 OOM。

**解决方案**: 在 FSDP2 环境下跳过 float 转换，保持 bfloat16 精度：

```python
if logits is not None and cross_entropy_fn.__name__ != "eager_cross_entropy":
    try:
        if get_parallel_state().fsdp_enabled:
            # FSDP2: keep bfloat16 to avoid OOM with large sequences
            pass
        else:
            logits = logits.float()
    except Exception:
        # If get_parallel_state fails, use float for safety
        logits = logits.float()
```

### 6. HuggingFace Backend Loss 返回格式兼容性 (`cross_entropy/__init__.py`)

**位置**: `veomni/ops/kernels/cross_entropy/__init__.py`

**问题**: HuggingFace 原始代码期望 loss 函数返回标量，但 VeOmni wrapper 返回元组。

**解决方案**: 检测调用模式并返回相应格式：

```python
# HuggingFace compatibility: when called without hidden_states/weights (HF calling pattern),
# return just the loss tensor instead of (loss, logits, fused_linear_aux) tuple.
if hidden_states is None and weights is None:
    # HF calling pattern - return just the loss tensor
    return loss

# VeOmni calling pattern - return (loss, logits, fused_linear_aux) tuple
return loss, logits, None
```

### 7. Liger Kernel Fallback 兼容性 (`liger.py`)

**位置**: `veomni/ops/kernels/cross_entropy/liger.py`

**问题**: `fused_liger_kernel_cross_entropy` 需要 `hidden_states` 和 `weights` 参数，但在 HuggingFace backend 下这些参数不可用。

**解决方案**: 添加 fallback 到 eager mode 并警告：

```python
def fused_liger_kernel_cross_entropy(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
):
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    if hidden_states is None or weights is None:
        # Fallback to eager mode
        import warnings
        warnings.warn("fused_liger_kernel_cross_entropy requires `hidden_states` and `weights`")
        from .eager import eager_cross_entropy
        return eager_cross_entropy(
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
            num_items_in_batch=num_items_in_batch,
            ignore_index=ignore_index,
            shift_labels=shift_labels,
            **kwargs,
        )
    return liger_kernel_cross_entropy(weights, hidden_states, labels), logits
```

## 已知问题

### FSDP2 + Gradient Checkpointing DTensor 错误

**症状**: `RuntimeError: aten.mul.Tensor: got mixed torch.Tensor and DTensor`

**原因**: 在 FSDP2 下使用 gradient checkpointing 时，重计算阶段的输入是普通 Tensor 而参数是 DTensor。

**当前状态**: 通过禁用 gradient checkpointing 绕过（`enable_gradient_checkpointing: false`）。

**建议解决方案**:
1. 升级到 PyTorch 2.6+（可能有更好的 DTensor 支持）
2. 修改模型实现以正确处理 DTensor

### HuggingFace Backend + Liger Kernel 限制

**症状**: Liger fused linear CE 无法在 HuggingFace backend 下启用，会 fallback 到 eager mode。

**原因**: HuggingFace 的 forward 方法不传递 `hidden_states` 和 `weights` 给 loss 函数。

**当前状态**: 会自动 fallback 到 eager mode，警告用户。

### 长序列 OOM 问题

**症状**: 使用大序列长度训练时可能出现 OOM。

**原因**: 
- logits tensor [batch, seq_len, vocab_size] 在 float32 下消耗显存是 bfloat16 的 2 倍
- transformers 某些版本强制 float32 转换

**当前状态**: 已添加 FSDP2 检测，跳过 float 转换。

**建议解决方案**:
1. 禁用 gradient checkpointing（但会增加激活显存）
2. 启用 FSDP2 CPU offload
3. 减少序列长度或 batch size

## 使用说明

### MODELING_BACKEND=hf 模式

当使用自定义模型需要设置 `MODELING_BACKEND=hf` 时：
1. 确保配置文件中的 `global_batch_size` 是实际 GPU 数量的倍数
2. HuggingFace backend 下某些优化 kernel 会自动 fallback 到 eager mode
3. Loss 返回格式会自动处理为兼容 HuggingFace 原始代码

### 配置文件格式

支持两种配置格式：

**嵌套格式 (VeOmni 标准)**:
```yaml
train:
  accelerator:
    fsdp_config:
      offload: true
```

**扁平格式 (兼容性支持)**:
```yaml
train:
  enable_fsdp_offload: true
```

两种格式都会被正确解析。

## 维护说明

添加新的兼容性补丁时：
1. 仅记录 VeOmni 框架本身的兼容性问题
2. 项目特定的功能（如教师-学生训练、自定义模型架构）应在项目文档中记录
3. 使用相对路径（如 `veomni/ops/__init__.py`）而非绝对路径
4. 添加适当的注释说明兼容性问题
5. 尽量使用运行时检测而非硬编码版本号
6. 保持向后兼容性
7. 避免包含项目特定路径、模型名称或私密信息
