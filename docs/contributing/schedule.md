# vLLM v1 调度算法详细分析

## 一、核心设计思想

vLLM v1 **没有传统的"prefill阶段"和"decode阶段"分离**。调度器采用基于token计数的统一模型：

> "每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`。每一步调度器尝试为请求分配token，使其 `num_computed_tokens` 追上 `num_tokens_with_spec`。这足以覆盖chunked prefill、prefix caching、speculative decoding和jump decoding优化。"

— `vllm/v1/core/sched/scheduler.py:310-320`

核心判断：`num_computed_tokens < num_prompt_tokens` → prefill；`num_computed_tokens >= num_prompt_tokens` → decode（每次只需1个token）。

---

## 二、调度算法两阶段流程

### Phase 1：调度 RUNNING 请求（`scheduler.py:346-514`）

遍历 `self.running` 列表，为已在运行的请求分配token：

```
对每个running请求:
  1. 计算 num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
     - decode请求 → num_new_tokens = 1（或1+spec_tokens）
     - prefill请求 → num_new_tokens = 剩余prompt token数
  2. 应用上限：
     - long_prefill_token_threshold（默认max_model_len*4%）
     - token_budget（全局token预算）
     - max_model_len（模型最大长度）
  3. 如果 num_new_tokens == 0 → continue（跳过，不break，让低优先级请求仍可调度）
  4. 分配KV blocks → kv_cache_manager.allocate_slots()
     - 分配失败 → 抢占最低优先级的running请求：
       * PRIORITY策略：抢占 (priority, arrival_time) 最大的请求
       * FCFS策略：抢占running列表最后一个请求
     - 抢占操作：释放KV blocks，重置 num_computed_tokens=0，prepend回waiting队列
     - 若抢占的请求就是当前请求 → break
  5. 成功 → 记录scheduled tokens、blocks、扣减budget
```

### Phase 2：调度 WAITING 请求（`scheduler.py:526-805`）

**仅在Phase 1无抢占且scheduler未暂停时执行**：

```
while (waiting or skipped_waiting) and token_budget > 0 and len(running) < max_num_seqs:
  1. 选择队列（FCFS优先skipped_waiting；Priority比较两队列头部）
  2. 尝试提升阻塞状态（grammar/KV/streaming就绪）
  3. LoRA约束检查（max_loras）
  4. Prefix cache查找：
     - 本地：kv_cache_manager.get_computed_blocks()
     - 远程：KVConnector查找外部缓存
  5. 计算 num_new_tokens = num_tokens - num_computed_tokens
  6. 应用上限（threshold、budget、chunked prefill）
     - 禁用chunked prefill且超预算 → break
  7. 分配KV blocks → 失败则break（不抢占running请求）
  8. 移入running，设status=RUNNING
```

**关键规则**：WAITING请求不会抢占RUNNING请求——内存不足时停止调度新请求。

---

## 三、抢占策略（Preemption）

vLLM v1 **不做CPU swap**，而是直接释放KV blocks并重置 `num_computed_tokens=0`：

```python
# scheduler.py:910-930
def _preempt_request(self, request, timestamp):
    kv_cache_manager.free(request)       # 释放所有KV blocks
    request.num_computed_tokens = 0      # 完全重置
    request.status = RequestStatus.PREEMPTED
    self.waiting.prepend_request(request) # 插回waiting队列头部
```

被抢占的请求重新调度时从头计算所有tokens（可受益于prefix caching）。

---

## 四、Prefill 执行流程

### 1. 请求到达 → 调度

```
AsyncLLM.generate() → InputProcessor.process_inputs() → EngineCoreRequest
→ Scheduler.add_request() → Request(num_computed_tokens=0, status=WAITING)
→ Scheduler.schedule() Phase 2:
   - 查找prefix cache hits
   - num_new_tokens = num_prompt_tokens - num_computed_tokens
   - 如果开启chunked prefill且超过threshold → 截断为threshold值
   - 分配KV blocks
   - 设 is_prefill_chunk = True
```

### 2. 模型执行

```
GPUModelRunner.execute_model():
  - add_requests(): 初始化req_states（prompt_len, all_token_ids, num_computed_tokens=0）
  - prepare_inputs():
    * is_prefilling_np = num_computed_prefill_tokens < prefill_len
    * prepare_prefill_inputs(): Triton kernel，将prompt token_ids[num_computed:num_computed+query_len]拷入input_ids
    * prepare_pos_seq_lens(): positions = num_computed_tokens + offset, seq_len = num_computed + query_len
  - build_attn_metadata(): CommonAttentionMetadata含query_start_loc、seq_lens、block_table、slot_mapping
  - model forward pass: attention kernel写KV到slot_mapping位置
```

### 3. 采样（Chunked Prefill特殊处理）

Chunked prefill期间 `num_sampled=0`（Triton kernel `_get_num_sampled_and_rejected_kernel` 显式置零），**不产出任何token**，仅填充KV cache。

最后一个prefill chunk时，logits在最后一个prompt token位置计算，产出第一个decode token。

### 4. Prefill → Decode 过渡

**自然过渡**，无显式状态切换：
- `num_computed_tokens += num_scheduled_tokens` 后 ≥ `num_prompt_tokens`
- `is_prefill_chunk = False`
- 下一步调度 `num_new_tokens = 1` → decode

---

## 五、Decode 执行流程

### 1. 调度

```
Scheduler.schedule() Phase 1:
  - decode请求: num_new_tokens = 1（或1+spec_tokens）
  - allocate_slots(): 分配1个（或1+lookahead）新KV slot
  - 若内存不足 → 抢占低优先级running请求
```

### 2. 批次重排

`reorder_batch_to_split_decodes_and_prefills()` 将batch分成4个区域：
1. **decode**: `num_scheduled ≤ threshold 且 done_prefilling`（放最前，适配CUDA graph）
2. **short_extend**: `num_scheduled ≤ threshold 且 still_prefilling`
3. **long_extend**: `num_scheduled > threshold 且 chunked prefilling`
4. **prefill**: `num_computed == 0`

### 3. 输入准备

```
_prepare_inputs():
  - positions = num_computed_tokens（已生成的位置）
  - input_ids = 上一步采样的token id
  - logits_indices = query_start_loc[1:] - 1（每个请求的最后一个query位置）
  - num_sampled_tokens = 1（每请求采样1个token）
  - block_table.compute_slot_mapping(): 将新token位置映射到物理KV slot
```

### 4. 注意力执行

FlashInfer对decode使用 `BatchDecodeWithPagedKVCacheWrapper`：
- 读取全序列的KV（通过paged_kv_indptr/indices/last_page_len）
- 写入新token的KV到slot_mapping指定位置
- query_len=1，高效批量decode

### 5. 采样与输出

```
sample_tokens():
  - 无spec: sampler(logits, sampling_metadata) → 每请求1个token
  - 有spec: rejection_sampler → 验证draft tokens，接受匹配的，拒绝不匹配的
  - propose_draft_token_ids(): 运行drafter（ngram/eagle/medusa等）提出下一步draft tokens
```

### 6. 输出更新

```
Scheduler.update_from_output():
  - append_output_token_ids(new_token)
  - check_stop(): 检查是否达到EOS/max_tokens
  - spec decode: 调整num_computed_tokens（减去rejected draft tokens数）
  - 完成的请求 → free KV blocks, 加入finished_req_ids
```

---

## 六、Prefill 与 Decode 使用的 Attention 算子

### 主要后端：FlashInfer（默认）

| 阶段 | 算子 | 代码位置 |
|------|------|---------|
| **Prefill** | `BatchPrefillWithPagedKVCacheWrapper.run()` | `flashinfer.py:1572` |
| **Prefill (DCP)** | `BatchDCPPrefillWrapper.run()`（内部：context→`BatchPrefillWithPagedKVCacheWrapper`, new_tokens→`BatchPrefillWithRaggedKVCacheWrapper`, merge→`merge_attn_states`) | `flashinfer.py:1536` |
| **Prefill (TRTLLM)** | `trtllm_batch_context_with_kv_cache()` | `flashinfer.py:1664` |
| **Decode** | `BatchDecodeWithPagedKVCacheWrapper.run()` | `flashinfer.py:1739` |
| **Decode (DCP)** | `BatchDecodeWithPagedKVCacheWrapper.run()` + `dcp_combine()` | `flashinfer.py:1723` |
| **Decode (TRTLLM)** | `trtllm_batch_decode_with_kv_cache()` | `flashinfer.py:1802` |
| **Cascade** | `MultiLevelCascadeAttentionWrapper.run()` | `flashinfer.py:1474` |

**选择逻辑**：`vllm/utils/flashinfer.py:336` 中的 `use_trtllm_attention()` 决定是否使用TRTLLM内核。条件包括平台支持、GQA可整除性、FP8量化等。默认自动模式：prefill在`kv_cache_dtype=="auto"`时用TRTLLM；decode在`num_tokens<=256`且`kv_cache_dtype=="auto"`时用TRTLLM。

### 其他后端的算子

| 后端 | Prefill 算子 | Decode 算子 |
|------|-------------|-------------|
| **FlashAttention** | `flash_attn_varlen_func()`（统一，不区分prefill/decode） | 同上 |
| **Triton** | `unified_attention()` 2D模式（grid: `(q_blocks, kv_heads)`） | `unified_attention()` 3D模式（grid: `(q_blocks, kv_heads, softmax_segments)`） |
| **CPU** | `F.scaled_dot_product_attention()` (PyTorch SDPA) | `ops.cpu_attention_with_kv_cache()` |
| **ROCm** | `chunked_prefill_paged_decode()` | `PagedAttention()` |
| **ROCm Aiter** | `rocm_aiter_ops.flash_attn_varlen_func` | `rocm_aiter_ops.batch_decode_with_paged_kv_cache` |
| **TurboQuant** | `flash_attn_varlen_func()` 或 `F.scaled_dot_product_attention()` | `triton_turboquant_decode_attention()` |
| **FlexAttention** | `flex_attention()`（torch.compile，统一） | 同上 |
| **Tree Attention** | `unified_attention()` | `unified_attention()` + tree_attn_bias |

### 核心区别

**Prefill 算子**：处理多token查询（query_len > 1），需计算整个prompt与KV cache的注意力，使用 varlen/paged prefill 内核。

**Decode 算子**：处理单token查询（query_len = 1），读取全序列paged KV cache生成1个新token，使用专用的decode内核（如FlashInfer的`BatchDecodeWithPagedKVCacheWrapper`），更轻量、支持CUDA graph优化。

FlashAttention 后端是个例外——用**同一个** `flash_attn_varlen_func()` 处理prefill和decode，依赖内核内部的varlen机制区分。

---

## 七、如何区分请求要做 Prefill 还是 Decode

vLLM v1 区分 prefill 和 decode 的核心判断在**三个层面**，全部基于同一个原理：**比较 `num_computed_tokens` 与 `num_prompt_tokens`**。

### 1. 调度器层面（`scheduler.py:946`）

```python
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```

`num_computed_tokens < num_prompt_tokens` → prefill chunk；否则进入 decode。请求创建时 `num_computed_tokens=0`（`request.py:145`），每步调度后递增，自然过渡。

### 2. Model Runner 层面（`model_runner.py:790`）

```python
is_prefilling_np = (
    self.req_states.num_computed_prefill_tokens[idx_mapping_np]
    < self.req_states.prefill_len.np[idx_mapping_np]
)
```

以及 `_build_attention_metadata()` 中（`model_runner.py:2200`）：

```python
is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
```

这两个布尔数组随请求传入 `CommonAttentionMetadata.is_prefilling`，传给各 attention 后端。

### 3. Attention 后端层面（`utils.py:606-683`）

`reorder_batch_to_split_decodes_and_prefills()` 基于**两个维度**将请求分成4类：

| 类别 | 条件 | 含义 |
|------|------|------|
| **decode** | `has_context ∧ below_threshold ∧ done_prefilling` | `num_computed ≥ num_prompt`，已进入decode |
| **short_extend** | `has_context ∧ below_threshold ∧ ~done_prefilling` | chunked prefill中间段，scheduled tokens ≤ threshold |
| **long_extend** | `has_context ∧ ~below_threshold` | chunked prefill中间段，scheduled tokens > threshold |
| **prefill** | `~has_context` (`num_computed==0`) | 首次prefill |

其中核心判断 `done_prefilling`：
```python
done_prefilling = num_computed_tokens_np >= num_prompt_tokens_np
```

之后 `split_decodes_and_prefills()`（`utils.py:507-576`）在排好序的batch上，通过 **`query_len > decode_threshold`** 找到 prefill/decode 边界：

```python
is_prefill = query_lens > decode_threshold   # 默认threshold=1
```

decode请求 `query_len=1`（每步只算1个token），prefill请求 `query_len>1`（一次算多个prompt token），所以**query长度本身就是最直接的区分标志**。

### 总结

| 层面 | 判断方式 | 本质 |
|------|---------|------|
| 调度器 | `num_computed_tokens < num_prompt_tokens` | token计数比较 |
| Model Runner | `num_computed_prefill_tokens < prefill_len` | 同上，GPU侧追踪 |
| Attention后端 | `query_len > 1`（即 `decode_threshold`） | **query长度**——prefill算多个token,decode算1个token |

三者本质一致：**prefill阶段还没算完prompt tokens，所以每步需要计算多个token（query_len>1）；decode阶段已算完prompt，每步只需1个新token（query_len=1）**。

---

## 八、关键配置参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `max_num_batched_tokens` | 2048 | 单步最大token数 |
| `max_num_seqs` | 128 | 单步最大序列数 |
| `long_prefill_token_threshold` | max_model_len×4% | 长prompt chunked prefill上限 |
| `enable_chunked_prefill` | True | 是否启用分块prefill |
| `policy` | "fcfs" | 调度策略（fcfs/priority） |
| `scheduler_reserve_full_isl` | True | 入场前检查完整序列能否放入KV cache |

---

## 九、整体流程图

```
EngineCore.step():
  1. Scheduler.schedule()
     ├── Phase 1: RUNNING → 分配tokens → KV block分配 → 可能抢占
     ├── Phase 2: WAITING → prefix cache查找 → chunked prefill → KV block分配 → 移入running
     └── 构建SchedulerOutput
  2. ModelExecutor.execute_model(scheduler_output)
     ├── 更新InputBatch（add/remove/condense/reorder）
     ├── prepare_inputs（prefill: Triton kernel拷token; decode: 取上次采样token）
     ├── build_attn_metadata（FlashInfer decode/prefill wrapper）
     ├── model forward（写KV到cache）
     ├── sample_tokens（prefill chunk: num_sampled=0; decode: 采样1 token; spec: rejection+draft）
  3. Scheduler.update_from_output()
     ├── append tokens → check_stop
     ├── spec: 调整num_computed_tokens
     ├── free finished requests
```
