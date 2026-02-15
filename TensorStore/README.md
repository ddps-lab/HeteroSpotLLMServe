# TensorStore

TensorStore downloads model weights from HuggingFace, partitions them for tensor parallelism, saves them in a custom raw binary format (TRAW), and uploads the results to S3. At serving time, a store server downloads the pre-partitioned tensors from S3, loads them onto GPUs, and exposes them to vLLM workers via a shared-memory manager.

## Quick Start: Uploading a Model

Edit `upload_model.sh` and fill in the two variables:

```bash
BUCKET_NAME="s3://your-bucket-name"
MODEL_NAME="your-model-name(currently support only text llama family, e.g., meta-llama/Llama-3.1-70B)"
```

Then run:

```bash
bash upload_model.sh
```

This will download the model from HuggingFace, partition every tensor for TP sizes 1, 2, 4, and 8, convert each partition to the TRAW binary format, and upload them all to S3.

## Why a Custom Binary Format?

PyTorch's `torch.save()` has a critical bug when saving sliced tensors, which caused significant issues in our experiments. When a tensor is sliced, the resulting view still references the original tensor's underlying storage. Calling `torch.save()` on this slice serializes the **entire original storage**, not just the visible slice.

What makes this bug particularly deceptive is that the tensor **contents** are correctly sliced — inspecting the loaded tensor shows the expected shape and values. Only the **file size** on disk reflects the full original storage.

**Example:**

Consider a 2 GB tensor of shape `(64128, 8192)` in bfloat16.
To apply tensor parallelism with TP=4, we slice it into 4 shards, each of shape `(16032, 8192)` (~500 MB).

| Step | Expected | Actual (`torch.save`) |
|------|----------|-----------------------|
| Save to file | 500 MB file | **2 GB file** (serializes full storage) |

The saved files look correct when loaded — the tensor shape and values are properly sliced. But each shard file is 2 GB instead of 500 MB. When the store server downloads these shards from S3 at initialization time, it transfers 4 x 2 GB = **8 GB** instead of the expected 4 x 500 MB = **2 GB**, creating a 4x network overhead that becomes a severe bottleneck, especially for large models partitioned across many GPUs.

To eliminate this overhead, TensorStore uses a custom raw binary format called **TRAW** that writes only the actual tensor data bytes, guaranteeing file size equals exactly `header_size + numel * element_size`.

## TRAW Binary Format Specification

### Overview

The TRAW format stores a single tensor per file with a compact header followed by raw data bytes. All multi-byte integers are **little-endian**.

### File Layout

```
[HEADER] [TENSOR DATA]
```

### Header Structure

| Offset | Size (bytes) | Type | Field | Description |
|--------|-------------|------|-------|-------------|
| 0 | 4 | char[4] | magic | Magic number `TRAW` (0x54524157) |
| 4 | 4 | uint32 | version | Format version (currently `1`) |
| 8 | 4 | uint32 | ndim | Number of dimensions (N) |
| 12 | 8 * N | uint64[] | shape | Shape array (one uint64 per dimension) |
| 12 + 8N | 4 | uint32 | dtype_code | Data type code (see table below) |
| 16 + 8N | 8 | uint64 | data_size | Expected data size in bytes (checksum) |
| 24 + 8N | data_size | bytes | data | Raw tensor data |

### Data Type Codes

| Code | PyTorch dtype | Element size |
|------|--------------|-------------|
| 1 | `torch.float16` | 2 bytes |
| 2 | `torch.bfloat16` | 2 bytes |
| 3 | `torch.float32` | 4 bytes |
| 4 | `torch.float64` | 8 bytes |

### Concrete Example

For a tensor of shape `(64128, 8192)` with dtype `bfloat16`:

```
Offset  0: b'TRAW'           # magic (4 bytes)
Offset  4: 1                 # version (4 bytes)
Offset  8: 2                 # ndim = 2 (4 bytes)
Offset 12: 64128             # shape[0] (8 bytes)
Offset 20: 8192              # shape[1] (8 bytes)
Offset 28: 2                 # dtype_code for bfloat16 (4 bytes)
Offset 32: 1050673152        # data_size = 64128 * 8192 * 2 (8 bytes)
Offset 40: [raw bytes...]    # 1,050,673,152 bytes of tensor data
```

Total file size: **40 bytes header + 1,050,673,152 bytes data = 1,050,673,192 bytes** (~1.0 GB)

Compared to `torch.save()`, which would produce a ~2.0 GB file for the same tensor due to the storage bug described above.

### Why uint8 View?

NumPy does not natively support `bfloat16`. To serialize a bfloat16 tensor to raw bytes:

1. **Save**: View the tensor as `torch.uint8` (reinterpretation, not conversion), convert to NumPy, then call `.tobytes()`.
2. **Load**: Read raw bytes into a `torch.uint8` tensor, then `.view(original_dtype).reshape(shape)`.

This is a zero-copy reinterpretation of the memory layout. No data is lost or transformed.

### Validation

The format includes multiple integrity checks:

- **Magic number**: Must be `b'TRAW'` (detects wrong file format)
- **Version**: Must be `1` (forward compatibility)
- **Dimension sanity**: `0 < ndim <= 10`, all dimensions > 0
- **Data size checksum**: The `data_size` field in the header must match `numel * element_size` computed from shape and dtype
- **File size verification** (on save): Written file size must equal `header_size + data_size`

## S3 Storage Structure

All TRAW files are stored under the `raw/` prefix to distinguish them from legacy `torch.save()` files.

```
s3://<bucket>/
  raw/
    <base_path>/
      <model_name>/
        config.json
        TP1/
          shard0/
            model.embed_tokens.weight.bin
            model.layers.0.self_attn.q_proj.weight.bin
            model.layers.0.self_attn.k_proj.weight.bin
            ...
        TP2/
          shard0/
            ...
          shard1/
            ...
        TP4/
          shard0/ ... shard3/
        TP8/
          shard0/ ... shard7/
```

Each `TP{size}/shard{rank}/` directory contains the complete set of tensors pre-partitioned for that specific tensor-parallel configuration, so the store server can load them directly without any runtime slicing.

## Components

### `raw_s3_model_uploader.py` (Uploader)

Downloads a model from HuggingFace, partitions each tensor for all requested TP sizes, saves in TRAW format, and uploads to S3.

**Tensor Partitioning Strategy:**

| Tensor type | Layers | Partition method |
|------------|--------|-----------------|
| Vocabulary/Embedding | `embed_tokens`, `lm_head` | Split along vocabulary dimension (padded to 64-token boundary) |
| Column-wise | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` | Split along output dimension |
| Row-wise | `o_proj`, `down_proj` | Split along input dimension |
| Other | `layernorm`, `input_layernorm`, etc. | Replicated (no split) |

### `raw_s3_tensor_store_server.py` (Store Server)

Downloads pre-partitioned TRAW tensors from S3, loads them onto the GPU, fuses related tensors, allocates KV cache, and serves them to vLLM workers via `multiprocessing.managers`.

**Tensor Fusion (after loading):**

- **QKV fusion**: `q_proj`, `k_proj`, `v_proj` are concatenated into a single `qkv_proj` tensor per layer.
- **Gate-Up fusion**: `gate_proj` and `up_proj` are concatenated into a single `gate_up_proj` tensor per layer.
