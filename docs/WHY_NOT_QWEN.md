# Why `kimi-k3-in-c` Cannot Run Qwen3.6-27B

*A case study in the difference between an inference **runtime** and an inference **implementation**.*

This document records a real question asked during a working session on an Apple Silicon
machine (M5 Max, 128 GB unified memory):

> I don't have Kimi K3 downloaded yet but I have Qwen3.6-27B here. Can we try running it
> with this?

The answer is no. The interesting part is *why* — the reason is architectural, and
understanding it explains what this repository actually is.

Every claim below was verified against source and against the real Qwen checkpoint on
disk. Commands are included so you can re-run them.

---

## 1. The short answer

`bin/k3` is not a general-purpose runtime like `llama.cpp`, vLLM, or MLX. It is a
**hardcoded implementation of exactly one model**: Kimi K3, a 2.78-trillion-parameter
mixture-of-experts model.

The distinction matters:

| | Runtime (llama.cpp, vLLM, MLX) | Implementation (this repo) |
|---|---|---|
| Model support | Many architectures via a registry | One architecture, inlined |
| New model | Add a config entry / converter | Write new kernels |
| Tensor lookup | Dynamic, name-mapped | Exact literal names in C |
| Unknown layer type | Dispatch error, or fallback | *No such branch exists* |

Qwen3.6-27B is a different architecture. There is no configuration flag, converter, or
quantization setting that bridges that gap, because the gap is in the compiled kernels.

---

## 2. What the engine hardcodes

### 2.1 A two-way layer dispatch with no third branch

From `src/core/k3_ops.c:64-71`:

```c
int k3_is_mla(const K3Cfg *c, int layer)
{
    for (int i = 0; i < c->n_full_attn; i++)
        if (c->full_attn[i] == layer + 1) return 1;
    return 0;
}
int k3_is_kda(const K3Cfg *c, int layer)   { return !k3_is_mla(c, layer); }
```

Read the second line carefully. KDA is defined as **`!is_mla`** — the logical negation.
There is no "unknown architecture" case, because the type system of this engine admits
exactly two kinds of layer. Any layer that is not Gated MLA is *by definition* treated as
Kimi Delta Attention, and the binder will then demand KDA's tensors from it.

This is a deliberate design choice, not an oversight. `include/k3/k3.h` states the target
precisely:

```
 *   93 layers: 69 Kimi Delta Attention (KDA) + 24 Gated MLA, plus one dense layer.
 *   Hidden 7168, 96 heads, 896 routed experts with top-16 selection and 2 shared,
 *   latent width 3584, SiTU-GLU activation, MXFP4 expert weights.
```

### 2.2 Tensor names as string literals

`src/model/k3_bind.c` requests weights by exact name, built with `snprintf`:

```c
reqn(p, &b->mla.q_a,  (int64_t)c->q_lora * H, PRE "layers.%d.self_attn.q_a_proj.weight", L);
reqw(p, &b->mla.q_a_norm, c->q_lora, -1, PRE "layers.%d.self_attn.q_a_layernorm.weight", L);
reqn(p, &b->mla.kv_a, ..., PRE "layers.%d.self_attn.kv_a_proj_with_mqa.weight", L);
...
reqw(p, &b->kda.q_conv, P * c->conv_k, -1, PRE "layers.%d.self_attn.q_conv1d.weight", L);
```

These names are compiled into the binary. A checkpoint that does not contain
`kv_a_proj_with_mqa` cannot satisfy the request, and there is no aliasing layer.

### 2.3 The engine fails *before* computing anything

This is the reassuring part. From `plan_resolve()` in `src/model/k3_bind.c:70-97`:

```c
/* Resolve and validate everything, and lay out the blob, before reading a byte. */
static int64_t plan_resolve(Plan *p, const K3St *s)
{
    ...
        q->t = k3_st_find(s, q->name);
        if (!q->t) {
            fprintf(stderr, "k3_bind: missing tensor %s\n", q->name);
            p->bad++;
            continue;
        }
        const int64_t have = k3_st_numel(q->t);
        /* The check that earns its keep: a shape the engine did not expect means the
         * config and the checkpoint disagree, and every kernel downstream would read
         * the wrong strides while producing plausible numbers. */
        if (q->want >= 0 && have != q->want) { ... p->bad++; continue; }
```

Two properties worth noting for anyone studying loader design:

1. **Fail-fast, pre-read.** Names *and* element counts are validated before a single
   weight byte is loaded.
2. **Shape checking is treated as load-bearing**, and the comment says why: a silently
   wrong shape yields *plausible numbers*, which is far worse than a crash.

So pointing `k3` at Qwen does not produce garbage text. It refuses and exits non-zero.
**The failure mode is a clean refusal, not silent corruption.** Section 7 shows the
actual terminal output of all three refusal gates.

---

## 3. What Qwen3.6-27B actually is

Measured from the local checkpoint
(`unsloth/Qwen3.6-27B-MLX-8bit`), not from memory:

```bash
python3 -c "
import json; c=json.load(open('config.json')); t=c.get('text_config',c)
print(c['model_type'], c['architectures'])
for k in ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim']:
    print(k, t.get(k))"
```

```
model_type      : qwen3_5
architectures   : ['Qwen3_5ForConditionalGeneration']
hidden_size     : 5120
num_hidden_layers: 64
num_attention_heads: 24
num_key_value_heads: 4     <- GQA: 24 query heads share 4 KV heads
head_dim        : 256
vocab_size      : 248320
quant bits/group: 8 / 64 mode: affine
```

Every K3-specific configuration key is **absent**:

```
  linear_attn_config  : absent      full_attn_layers    : absent
  kda_layers          : absent      num_experts         : absent
  n_routed_experts    : absent      q_lora_rank         : absent
  kv_lora_rank        : absent
```

`k3_cfg.h:168-172` requires `full_attn_layers` and calls `k3cfg_miss()` when it is
missing. The header even documents the consequence of proceeding without it
(`k3_cfg.h:24`):

```
 *     - full_attn_layers would come back empty, so k3_is_mla() answers 0 for every layer
```

Combined with `k3_is_kda() = !k3_is_mla()`, an empty layer map means **every layer is
classified KDA** — the engine would demand KDA tensors from all 64 Qwen layers. This is
precisely the silent-misconfiguration failure the config reader exists to prevent.

### 3.1 A genuinely interesting wrinkle

Qwen3.6-27B is **not** a plain transformer. Inspecting the tensor index reveals a
*hybrid* architecture:

```
linear_attn layers: 48
self_attn  layers : 16 -> [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
```

Every fourth layer is full attention; the other three are linear attention. And the
linear-attention layers carry:

```
layers.0.linear_attn.A_log
layers.0.linear_attn.conv1d.weight
layers.0.linear_attn.dt_bias
layers.0.linear_attn.in_proj_qkv.weight
```

`A_log`, `dt_bias`, `conv1d` — these are the **same family of gated-delta / state-space
primitives** that K3's KDA uses. Both models independently landed on the same high-level
idea: *mostly linear-attention layers, punctuated by periodic full-attention layers.*

Kimi K3: 69 KDA + 24 MLA (of 93).
Qwen3.6-27B: 48 linear + 16 full (of 64).

**This is what makes the incompatibility instructive rather than boring.** These two
models are conceptually cousins. They still cannot share an engine, because agreeing on
the *idea* is not the same as agreeing on the *realization*.

---

## 4. The structural mismatches

| Dimension | Kimi K3 (required) | Qwen3.6-27B (actual) | Bridgeable? |
|---|---|---|---|
| **Linear-attn tensors** | `q_conv1d`, `k_conv1d`, `v_conv1d`, `f_a_proj`, separate q/k/v/g | `conv1d`, fused `in_proj_qkv`, `in_proj_a/b/z` | No — different factorization |
| **Full-attn type** | Gated MLA: `q_a_proj`→`q_b_proj` LoRA, `kv_a_proj_with_mqa`, latent 3584 | GQA: plain `q_proj`/`k_proj`/`v_proj`, 24Q/4KV, `q_norm`/`k_norm` | No — MLA's latent KV cache has no GQA analogue |
| **FFN** | 896 routed experts, top-16 + 2 shared, SiTU-GLU | Dense `gate/up/down_proj` | No — engine's MoE router has no dense path |
| **Weight dtype** | MXFP4 (multiplied in packed form), bf16 trunk | MLX affine int8, `group_size` 64, `.scales`/`.biases` sidecars | No — different dequant math |
| **Layer count / width** | 93 layers, hidden 7168, 96 heads | 64 layers, hidden 5120, 24 heads | Config-only, but moot |
| **Tensor prefix** | `model.layers.N.…` | `language_model.model.layers.N.…` | Trivially, but moot |

The last two rows are the only cosmetic ones. Rows 1–4 are structural: each requires
different arithmetic, not different parameters.

Note especially the **MXFP4 vs. MLX-int8** row. `k3.h:12` explains that experts are
"multiplied straight out of their packed MXFP4 form, never widened to fp32" — the packed
format is fused into the matmul kernel as a performance necessity. Qwen's MLX affine
quantization stores separate `.scales` and `.biases` tensors with different group
semantics. Supporting it means a second dequantizing matmul kernel, not a flag.

---

## 5. What "make it work" would actually require

Not a converter. A new engine sharing only the I/O layer:

1. **A GQA attention kernel** — 24 query heads over 4 KV heads, with `q_norm`/`k_norm`.
   K3 has no such kernel; MLA's latent-cache design is not reducible to GQA.
2. **A second linear-attention kernel** for Qwen's fused `in_proj_qkv` layout, distinct
   from KDA's split projections and per-channel conv.
3. **An MLX affine-int8 dequant path** (`.scales`/`.biases`, group 64) alongside MXFP4.
4. **A dense-FFN path** bypassing the expert router, cache, and streaming machinery —
   which is to say, bypassing most of what this repo is *for*.
5. **A new binder** with Qwen's names, prefix, and layer map.
6. **A third layer-type branch**, breaking the `is_kda = !is_mla` invariant that the
   test-suite's oracle gates depend on.

At that point you have written a Qwen engine. The reusable parts — the safetensors
reader, the LRU cache, the streaming trunk — are the *I/O* layer, not the math.

Worth stressing: much of this engine's value **is** the streaming machinery, and that
machinery exists to solve a problem Qwen3.6-27B does not have. K3 is 1.56 TB and cannot
fit in RAM. Qwen3.6-27B at 8-bit is ~34.7 GB and fits comfortably in 128 GB with room to
spare. The expert cache, the trunk ring, the MXFP4-in-place matmul — all of it is
machinery for a model that *cannot* be resident. Pointing it at a model that fits is
using a crane to pick up a coffee cup.

---

## 6. Practical options

**To run Qwen3.6-27B** — use the existing vllm-mlx setup. Measured on this machine at
~15.4 tok/s (8-bit, 1k–8k context). This engine would not be faster even if it worked;
it optimizes for *capacity*, not throughput on models that already fit.

**To exercise this engine** — obtain the K3 checkpoint, but check the size first:
113.49 GB resident weights plus 1.45 TB of streamed experts. `k3.h` notes the model
"runs in 8 GB of RAM and in 224 GB, and produces byte-identical output at both" — the
streaming design means the *download*, not the RAM, is the binding constraint.

**To learn from it** — no checkpoint needed. `make test` passes standalone and includes a
full-model oracle that validates against a reference implementation:

```
GATE 1  teacher forcing : 32/32 positions match tf_pred
GATE 2  greedy decode   : 20/20 generated tokens match full_ids
GATE 3  incremental     : 20/20 generated tokens match full_ids  <- KV cache + carried KDA state
VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

This is arguably the repository's main contribution: a 143 KB README and a test-suite
that pins down five architectural invariants, each documented as a place where "a
plausible-looking implementation produces a model that runs, emits fluent text, and is
wrong."

---

## 7. The actual terminal output

Everything above is read off the source. This section is what the binary *actually
prints*, captured on 2026-08-03. There are **three independent refusal gates**, and the
engine never reaches arithmetic at any of them.

### Gate 1 — the config reader

The honest attempt: point `k3` at the real Qwen snapshot.

```bash
QDIR=$(ls -d ~/models/huggingface/hub/models--unsloth--Qwen3.6-27B-MLX-8bit/snapshots/*/)
./bin/k3 "$QDIR" --ids 1,2,3
```

```
k3_cfg: .../config.json is missing 19 required field(s):
    num_heads
    short_conv_kernel_size
    gate_lower_bound
    q_lora_rank
    kv_lora_rank
    qk_nope_head_dim
    qk_rope_head_dim
    v_head_dim
    num_experts
    num_experts_per_token
    num_shared_experts
    routed_expert_hidden_size
    moe_intermediate_size
    routed_scaling_factor
    first_k_dense_replace
    attn_res_block_size
    activation_situ_beta
    activation_situ_linear_beta
    full_attn_layers
  refusing to substitute defaults: a config this reader cannot
  fully understand would silently produce a DIFFERENT model.
ABORTED: the model config could not be read with confidence.
```

Nineteen fields, named individually. Note the stated reason for refusing defaults: a
partially-understood config "would silently produce a DIFFERENT model." This is the same
philosophy as the shape check in §2.3 — the engine treats *plausible but wrong* as the
worst possible outcome.

**This is where a genuine attempt stops.** Everything below required deliberately
constructing inputs to defeat each gate, purely to observe the next one.

### Gate 2 — the safetensors dtype reader

Supplying a synthetic K3-shaped `config.json` (Qwen's real dimensions and layer map)
alongside symlinks to Qwen's actual weights gets past the config reader:

```
config: .../config.json (flat shape) | hidden=5120 layers=64 vocab=248320
        | 16 MLA + 48 KDA | experts 8 top2 shared2 | latent=2048
Kimi K3, pure C, released checkpoint
  shards   : .../qwen_as_k3
  prompt   : 3 tokens, generating 8

k3_st: model-00001-of-00007.safetensors: unsupported dtype 'U32'
       on language_model.model.embed_tokens.weight
```

Exit status 1.

Two things worth pausing on:

1. **`16 MLA + 48 KDA`** — the engine parsed the layer map and derived *exactly* Qwen's
   real 16/48 hybrid split (§3.1). The architectural resemblance is close enough that
   K3's own config reader describes Qwen's topology correctly. It still cannot run it.
2. **`unsupported dtype 'U32'`** — this is the quantization mismatch from the §4 table,
   made concrete. MLX packs int8 weights into `uint32` words with `.scales`/`.biases`
   sidecars. K3's reader handles BF16/F32/MXFP4 and has no `U32` case. The very first
   tensor it touches is already unreadable.

### Gate 3 — the tensor binder

To reach the binder, the dtype gate must also be defeated: a fixture of BF16 tensors
carrying *Qwen's* names, with the same synthetic config.

```
k3_bind: missing tensor language_model.model.layers.0.input_layernorm.weight
k3_bind: missing tensor language_model.model.layers.0.post_attention_layernorm.weight
k3_bind: missing tensor language_model.model.layers.0.self_attention_res_norm.weight
k3_bind: missing tensor language_model.model.layers.0.self_attention_res_proj.weight
k3_bind: missing tensor language_model.model.layers.0.mlp_res_norm.weight
k3_bind: missing tensor language_model.model.layers.0.mlp_res_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.q_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.k_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.v_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.g_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.o_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.q_conv1d.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.k_conv1d.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.v_conv1d.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.f_a_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.f_b_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.b_proj.weight
k3_bind: missing tensor language_model.model.layers.0.self_attn.A_log
k3_bind: missing tensor language_model.model.layers.0.self_attn.dt_bias
k3_bind: missing tensor language_model.model.layers.0.self_attn.o_norm.weight
k3_bind: missing tensor language_model.model.layers.0.mlp.gate_proj.weight
...
```

**1,692 missing tensors**, exit status 1.

This is the `is_kda = !is_mla` negation from §2.1 in action, and it is the single most
instructive line of output in this document. Layer 0 is not in `full_attn_layers`, so
`k3_is_mla()` returns 0, so `k3_is_kda()` returns 1 — and the engine demands **KDA**
tensors from it:

| Engine demands (K3 KDA) | Qwen actually ships |
|---|---|
| `layers.0.self_attn.A_log` | `layers.0.linear_attn.A_log` |
| `layers.0.self_attn.dt_bias` | `layers.0.linear_attn.dt_bias` |
| `layers.0.self_attn.q_conv1d.weight` | `layers.0.linear_attn.conv1d.weight` |
| `layers.0.self_attn.{q,k,v,g}_proj.weight` (split) | `layers.0.linear_attn.in_proj_qkv.weight` (fused) |
| `layers.0.self_attn.f_a_proj` / `f_b_proj` / `b_proj` | *(no equivalent)* |

Both models have `A_log`. Both have `dt_bias`. Both have a `conv1d`. They are the same
gated-delta primitives — under different module names, with different projection
factorizations, one split and one fused.

**That table is the entire thesis of this document in five rows.** The architectures are
close cousins. The engine still cannot bind a single tensor, because inference engines
are written against tensor layouts, not against family resemblance.

### Reproducing gates 2 and 3

Both synthetic configurations were built in a scratch directory using **symlinks** to the
Qwen weights; the model directory was verified unmodified afterward (13 files, unchanged).
Nothing here mutates a checkpoint.

```bash
SP=/tmp/k3_qwen_demo && mkdir -p "$SP/qwen_as_k3"
QDIR=$(ls -d ~/models/huggingface/hub/models--unsloth--Qwen3.6-27B-MLX-8bit/snapshots/*/)
ln -sf "$QDIR"*.safetensors "$SP/qwen_as_k3/"

# Gate 2: K3-shaped config carrying Qwen's real dimensions + layer map
python3 -c "
import json
full=[L+1 for L in range(3,64,4)]          # Qwen's real full-attn layers, ONE-based
json.dump({'hidden_size':5120,'num_hidden_layers':64,'vocab_size':248320,
 'rms_norm_eps':1e-06,'tie_word_embeddings':False,'kda_num_heads':24,'kda_head_dim':256,
 'short_conv_kernel_size':4,'gate_lower_bound':-5.0,'num_attention_heads':24,
 'q_lora_rank':1536,'kv_lora_rank':512,'qk_nope_head_dim':128,'qk_rope_head_dim':64,
 'v_head_dim':128,'mla_use_output_gate':True,'num_experts':8,'num_experts_per_token':2,
 'num_shared_experts':2,'routed_expert_hidden_size':2048,'moe_intermediate_size':2048,
 'routed_scaling_factor':1.0,'moe_renormalize':True,'latent_moe_use_norm':True,
 'first_k_dense_replace':1,'intermediate_size':17408,'attn_res_block_size':4,
 'situ_beta':4.0,'situ_linear_beta':25.0,'full_attn_layers':full},
 open('$SP/qwen_as_k3/config.json','w'),indent=1)"

./bin/k3 "$SP/qwen_as_k3" --ids 1,2,3      # -> unsupported dtype 'U32'
```

For gate 3, replace the shards with a small BF16 safetensors file using Qwen's tensor
names (five tensors suffice) and re-run; the binder then reports its 1,692 misses.

---

## 8. The generalizable lesson

> **Two models can share a design philosophy and still be unable to share an engine.**

Kimi K3 and Qwen3.6-27B both concluded that mostly-linear attention with periodic full
attention is a good idea. A block diagram of the two would look similar. Yet they differ
in projection factorization, normalization placement, quantization format, and FFN
topology — and inference engines are written against *those* details, not against block
diagrams.

This is why the ecosystem has converged on runtimes with architecture registries and
conversion tooling (GGUF, MLX, vLLM model classes). `kimi-k3-in-c` deliberately opts out
of that generality to make one model's mechanics legible in readable C. That is a
legitimate and useful trade — but the cost of it is exactly this: it runs Kimi K3, and
nothing else.

---

## Appendix: reproducing these findings

```bash
# The two-way dispatch with no third branch
sed -n '64,71p' src/core/k3_ops.c

# Hardcoded tensor names
grep -n 'self_attn\.' src/model/k3_bind.c | head -20

# Fail-fast validation before any weight is read
sed -n '70,97p' src/model/k3_bind.c

# Qwen's hybrid layer pattern (run in the snapshot dir)
python3 -c "
import json,glob
idx=json.load(open(glob.glob('*.safetensors.index.json')[0]))['weight_map']
names=set(idx)
lin =[L for L in range(64) if any(f'.layers.{L}.linear_attn.' in n for n in names)]
full=[L for L in range(64) if any(f'.layers.{L}.self_attn.'   in n for n in names)]
print('linear_attn:', len(lin), ' self_attn:', len(full), '->', full)"
```

Verified 2026-08-03 against commit `85ab2cd` and
`unsloth/Qwen3.6-27B-MLX-8bit`.
