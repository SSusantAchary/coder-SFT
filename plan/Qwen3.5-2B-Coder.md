# Qwen3.5-2B-Coder — SFT Plan

**Target model:** `unsloth/Qwen3.5-2B`  
**Target artifact:** `Qwen3.5-2B-Coder`  
**Primary goal:** Turn Qwen3.5-2B into a strong small coding assistant with repository-aware behavior and useful long-context performance up to **128K tokens**, while retaining instruction-following and tool-use ability.  
**Training method:** SFT with 16-bit BF16 LoRA first; full fine-tuning only if clearly justified by evaluation.  
**Primary stack:** Unsloth + Hugging Face Transformers + TRL + PEFT + Datasets  
**Deployment target:** Safetensors + GGUF for `llama.cpp`; optional vLLM/SGLang serving.

---

## 1. Executive recommendation

Do **not** train all examples at 128K.

Use a staged curriculum:

```text
Qwen3.5-2B
   │
   ├── Stage 0: Baseline evaluation
   │
   ├── Stage 1: Core coding SFT              4K–8K
   │
   ├── Stage 2: Code editing + debugging     8K–16K
   │
   ├── Stage 3: Repository engineering       16K–32K
   │
   ├── Stage 4: Long-context repository SFT  32K–64K
   │
   ├── Stage 5: 128K specialization          64K–128K
   │
   ├── Stage 6: Final evaluation + ablation
   │
   └── Stage 7: Merge → GGUF → llama.cpp
```

The model already has a native maximum position configuration of **262,144 tokens**, so the project should focus on teaching it to **use long code context**, not on extending its positional range.

The first serious version should use **16-bit BF16 LoRA**. Current Unsloth guidance advises against 4-bit QLoRA for Qwen3.5 because its quantization difference is higher than normal. A 2B model fits BF16 LoRA comfortably on the target 40GB/80GB Colab GPUs; activation memory at long context must still be measured before training.

The validated v1 implementation is in the parent `coder_SFT/` directory. It targets 8K core coding followed by a memory-gated 16K–32K repository stage. Training beyond 32K remains a later experiment rather than a v1 promise.

---

# 2. What “Coder” should mean

The goal should not be only:

> “Given a LeetCode-style question, produce code.”

The target behavior should cover six capability groups.

| Capability | Example |
|---|---|
| Code generation | Implement a function/class/module from requirements |
| Code understanding | Explain unfamiliar code and identify control/data flow |
| Debugging | Identify failure cause from code + traceback + tests |
| Code editing | Modify existing code while preserving interfaces |
| Repository reasoning | Locate relevant files across a large project |
| Tool-oriented engineering | Propose/run tests, inspect files, produce patches |

A useful 2B coder should learn this sequence:

```text
issue / request
      ↓
understand constraints
      ↓
locate relevant code
      ↓
reason about dependencies
      ↓
make minimal change
      ↓
write/update tests
      ↓
verify
      ↓
return patch or final code
```

The highest-value differentiator over a generic 2B model is therefore:

**small model + strong code editing + repository localization + long-context discipline.**

---

# 3. Base model

Use:

```text
unsloth/Qwen3.5-2B
```

rather than the GGUF package for training.

The model's released configuration includes:

- native context setting: 262,144 tokens
- 24 text layers
- hybrid linear/full-attention pattern
- MTP support in the architecture
- Apache-2.0 license on the Unsloth checkpoint
- tool-use / Qwen Code-oriented chat template support

The GGUF repository should be treated as a **deployment artifact**, not the main SFT source.

## Training → deployment flow

```text
unsloth/Qwen3.5-2B
        ↓
BF16 LoRA SFT
        ↓
adapter checkpoint
        ↓
merge into base model
        ↓
BF16 / FP16 Safetensors
        ↓
GGUF conversion
        ↓
Q4_K_M / Q5_K_M
        ↓
llama.cpp
```

---

# 4. Success criteria

Define success before training.

## 4.1 Functional coding quality

The SFT model should improve over the original Qwen3.5-2B on:

- HumanEval+ / EvalPlus
- MBPP+
- LiveCodeBench
- BigCodeBench
- internal coding tasks

Do not optimize only for HumanEval. It is too narrow.

## 4.2 Editing quality

Measure:

- correct files modified
- patch applies cleanly
- existing tests remain green
- newly added tests pass
- minimality of patch
- no unnecessary refactoring
- API compatibility

## 4.3 Repository quality

Measure:

```text
File localization Recall@1
File localization Recall@3
File localization Recall@5

Function localization Recall@k

Patch success rate
Test pass rate
Issue resolution rate
```

## 4.4 Long-context quality

At each context bucket:

```text
8K
16K
32K
64K
96K
128K
```

measure:

- relevant-file retrieval
- dependency understanding
- code-needle retrieval
- cross-file reasoning
- patch correctness
- hallucinated-file rate
- irrelevant-edit rate

The model should not merely accept 128K input. It should prove that useful information at token 100K+ changes the answer correctly.

---

# 5. Dataset strategy

Use a mixture instead of one dataset.

## Recommended training mixture

Use the already-filtered EER6 OpenCodeInstruct variants instead of starting from the raw 5M-example NVIDIA dataset for the first experiments.

| Source | Initial target | Purpose |
|---|---:|---|
| `EER6/nvidia-OpenCodeInstruct-refined` | 200K–300K | highest-quality coding foundation |
| `EER6/nvidia-OpenCodeInstruct-broad` | 100K–250K | diversity, harder cases, broader coverage |
| CommitPackFT | 50K–100K | natural-language code modifications |
| SWE-smith language datasets | 20K–50K | repository-level software engineering |
| Custom repo SFT | 10K–30K | file localization + long-context behavior |
| Tool/action traces | 5K–20K | inspect/edit/test workflow |
| Hard failure repairs | 5K–10K | debugging and self-correction |

Do not blindly maximize examples. For a 2B model, **quality, execution correctness, diversity, and task distribution matter more than adding millions of repetitive examples**.

### Recommended Stage-1 default

```text
EER6 OpenCodeInstruct Refined    250K   ~60%
EER6 OpenCodeInstruct Broad      150K   ~35%
Debug / special coding tasks      20K   ~5%
                                -----
Total                            ~420K
```

The first experiment can be much smaller; the ~420K mixture is the recommended serious Stage-1 run after the pipeline is validated.

---

# 6. Dataset A — OpenCodeInstruct Broad + Refined

Use these two datasets as the main coding-SFT source:

```text
EER6/nvidia-OpenCodeInstruct-broad
EER6/nvidia-OpenCodeInstruct-refined
```

They are filtered versions of NVIDIA OpenCodeInstruct and are more convenient for the first Qwen3.5-2B experiments than immediately processing the full raw 5M-example corpus.

## 6.1 Role of each dataset

### Refined = quality anchor

Use:

```text
EER6/nvidia-OpenCodeInstruct-refined
```

as the highest-confidence coding subset.

It is intended to retain examples with the strictest judge and execution criteria, making it suitable for the majority of Stage-1 supervision.

Recommended use:

```text
200K–300K examples
```

### Broad = diversity pool

Use:

```text
EER6/nvidia-OpenCodeInstruct-broad
```

to recover task diversity and somewhat harder/noisier cases while retaining strong quality filtering.

Recommended use:

```text
100K–250K examples
```

Do **not** start by training on the entire broad dataset. First establish whether 300K–500K carefully sampled examples improve the 2B model.

## 6.2 Why use both?

The desired trade-off is:

```text
Refined
    │
    ├── high confidence
    ├── stronger execution correctness
    └── lower noise

Broad
    │
    ├── more task diversity
    ├── more difficult examples
    └── broader coverage
```

Combined:

```text
quality anchor + diversity
             ↓
better Stage-1 coder mixture
```

## 6.3 Recommended Stage-1 mixture

A strong default:

```text
Refined OpenCodeInstruct     250K   60%
Broad OpenCodeInstruct       150K   35%
special/debug examples        20K    5%
                            -----
Total                       ~420K
```

A cheaper first experiment:

```text
Refined                       35K
Broad                         15K
Commit/edit/debug             10K
                             ----
Total                         60K
```

This is enough to validate:

- dataset formatting
- Qwen chat template
- LoRA target modules
- training stability
- code benchmark improvement
- merge/export correctness

before scaling.

## 6.4 Broad dataset stratification

Do not sample `broad` uniformly if quality metadata is available.

Construct quality tiers such as:

```text
Tier A
- perfect test execution
- strongest judge scores

Tier B
- perfect test execution
- judge scores >= 4

Tier C
- ~90% test score
- judge scores >= 4

Tier D
- ~80% test score
- judge scores >= 4
```

Suggested sample distribution from the broad pool:

```text
Tier A    50%
Tier B    30%
Tier C    15%
Tier D     5%
```

The intent is to preserve diversity without allowing borderline examples to dominate training.

## 6.5 Additional cleanup

Even filtered datasets should still be processed for:

- exact duplicates
- near-duplicate instructions
- near-duplicate solutions
- malformed code blocks
- truncated responses
- highly repetitive synthetic templates
- suspicious benchmark overlap
- extremely verbose explanations that add little code signal

Prefer a reproducible local manifest that records:

```text
source dataset
source row id
quality tier
language
token count
test score
judge score
dedup cluster
selected / rejected
```

## 6.6 Sampling by capability

Stage-1 should not become only algorithmic problem solving.

Where metadata or classification allows, preserve a mixture such as:

```text
code generation          35%
debugging                20%
code completion          10%
data structures/algo     15%
API/library usage        10%
reasoning/explanation     5%
tests/validation          5%
```

Exact percentages can change after profiling the dataset.

## 6.7 Optional raw NVIDIA dataset

Keep:

```text
nvidia/OpenCodeInstruct
```

as an upstream source for later research, not as the first training input.

Use the raw corpus only when you want to:

- reproduce the EER6 filtering yourself
- construct alternative quality thresholds
- add examples excluded by the community subsets
- study quality-vs-diversity ablations

For v1 and v2, the EER6 broad/refined split is the preferred route.

# 7. Dataset B — CommitPackFT

Use:

```text
bigcode/commitpackft
```

CommitPackFT is a filtered commit dataset where commit messages are shaped more like natural-language instructions.

Convert examples into:

```text
SYSTEM:
You are a software engineer. Make the smallest correct change.

USER:
Repository context:
<before version>

Task:
<commit message / instruction>

ASSISTANT:
<patch or updated code>
```

This stage teaches something OpenCodeInstruct does not teach well:

> modify existing code rather than always generating a solution from scratch.

Prefer examples where:

- diff is reasonably small
- commit message is descriptive
- change is coherent
- language is in the target language set
- generated context fits the current stage length

---

# 8. Dataset C — SWE-smith

Prefer the current **language-specific SWE-smith datasets**, for example:

```text
SWE-bench/SWE-smith-py
SWE-bench/SWE-smith-js
SWE-bench/SWE-smith-ts
SWE-bench/SWE-smith-java
SWE-bench/SWE-smith-go
SWE-bench/SWE-smith-cpp
SWE-bench/SWE-smith-rs
```

The Python split alone currently contains about 50K task instances across more than 100 repositories, and SWE-smith tasks include executable environments.

## Do not simply feed the raw task row to SFT

Construct useful training trajectories.

Example:

```text
USER
<issue>

ASSISTANT
I need to inspect:
- src/cache.py
- tests/test_cache.py

TOOL
<file contents>

ASSISTANT
The failure is caused by ...

I will modify ...

TOOL
<test result>

ASSISTANT
<final patch>
```

For plain SFT, the trajectory can be flattened into messages.

---

# 9. Custom repository dataset

This is the most important part for the 128K objective.

The model already understands generic coding. We need to teach:

```text
large repository context
+
issue
+
relevant information is sparse
+
correct files may occur very late in context
+
make a small verified patch
```

## 9.1 Construct examples from real repositories

For each task:

1. check out repository at commit N
2. capture issue/commit intent
3. identify changed files from commit N+1
4. build a pre-change repository snapshot
5. construct context
6. use the real diff as the target
7. run tests where feasible
8. attach verification metadata

## 9.2 Context construction

Do not always place the answer file at the beginning.

Randomize relevant-file position:

```text
0–20% of context
20–40%
40–60%
60–80%
80–100%
```

Otherwise the model learns a positional shortcut.

## 9.3 Add distractor files

A good 128K example should contain:

```text
relevant files      5K–15K
dependencies        10K–30K
tests               5K–20K
distractors         remainder
```

But distractors should be **realistic**, not random Wikipedia text.

Use:

- neighboring packages
- similarly named classes
- stale implementation paths
- unrelated tests
- generated files only in limited proportion

## 9.4 Long-context sample types

Create at least these categories:

### A. File localization

Input:
- repo tree
- selected file summaries/content
- issue

Target:
- ranked file list

### B. Function localization

Target:
- exact functions/classes likely requiring change

### C. Bug explanation

Target:
- root cause with evidence from files

### D. Patch generation

Target:
- unified diff

### E. Test generation

Target:
- regression tests

### F. End-to-end repair

Target:
- diagnosis + patch + tests

This decomposition is valuable because a 2B model may learn repository behavior better when the intermediate skills are explicitly supervised.

---

# 10. Dataset format

Use conversational or prompt-completion examples.

Recommended conceptual format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a precise software engineering assistant. Prefer minimal verified changes."
    },
    {
      "role": "user",
      "content": "Repository:\n...\n\nIssue:\n..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

For patch tasks:

```json
{
  "prompt": "...repository + issue...",
  "completion": "...unified diff..."
}
```

For TRL, prefer computing loss only on assistant/completion tokens where the template supports it.

---

# 11. Output style to teach

A 2B model should not waste capacity generating large amounts of generic prose.

Teach a compact engineering response:

```text
1. Diagnosis
2. Files affected
3. Patch
4. Tests
```

For agent-style data:

```text
inspect
→ reason
→ edit
→ test
→ repair if needed
→ final
```

Avoid training long hidden-style reasoning transcripts as a requirement. Prefer short observable engineering rationale and verified actions.

---

# 12. Train/validation/test split

Never randomly split individual examples from the same repository into train and test.

That creates leakage.

Use:

```text
repository-level split
```

Example:

```text
Train repositories       90%
Validation repositories   5%
Test repositories         5%
```

For real benchmark evaluation, keep benchmark repositories/tasks out of SFT whenever possible.

Also deduplicate against:

- HumanEval
- MBPP
- LiveCodeBench tasks
- BigCodeBench
- SWE-bench evaluation problems

---

# 13. Stage 0 — baseline benchmark

Before touching training:

```text
Qwen3.5-2B baseline
```

Run exactly the same evaluation harness that will be used later.

Record:

```text
HumanEval+ pass@1
MBPP+ pass@1
LiveCodeBench
BigCodeBench

internal generation pass rate
internal edit pass rate
file localization Recall@k

8K repo score
32K repo score
64K repo score
128K repo score

generation latency
tokens/sec
VRAM
```

Save all raw outputs.

This becomes:

```text
baseline.json
```

Without this stage, you cannot tell whether SFT improved coding behavior or merely changed response style.

---

# 14. Stage 1 — core coding SFT

## Objective

Improve:

- code generation
- correctness
- instruction following
- algorithmic implementation
- basic debugging

## Dataset

Use the EER6 OpenCodeInstruct variants:

```text
EER6/nvidia-OpenCodeInstruct-refined
EER6/nvidia-OpenCodeInstruct-broad
```

Recommended serious Stage-1 mixture:

```text
Refined        ~250K
Broad          ~150K
Other/debug     ~20K
```

For the first validation run, use only ~50K–60K examples.

## Context

```text
max_seq_length = 8192
```

Most examples will be considerably shorter.

## Starting BF16 LoRA settings

```text
16-bit BF16 LoRA
LoRA rank             32
LoRA alpha            32
LoRA dropout          0.0
gradient checkpoint   unsloth
optimizer             adamw_8bit
learning rate         1e-4 initially
warmup ratio          ~0.03
epochs                 1 initially
```

Treat these as starting values, not sacred constants.

Run a small learning-rate sweep:

```text
5e-5
1e-4
2e-4
```

Evaluate after a controlled token budget.

## Gate to Stage 2

Proceed only if:

- code benchmark score improves
- instruction following is retained
- no major verbosity/repetition regression
- validation loss is stable

---

# 15. Stage 2 — code editing and debugging

## Objective

Teach modification rather than greenfield generation.

Dataset mix:

```text
CommitPackFT                     ~50%
EER6 Broad/Refined rehearsal       ~25%
debug/fix examples                 ~25%
```

Context:

```text
8K–16K
```

Tasks:

- fix bug
- complete TODO
- modify API
- refactor minimally
- add unit test
- repair test failure
- optimize without behavior change

Output preferably:

```text
unified diff
```

or a strict edit format.

## Gate

Measure:

```text
patch applies %
tests pass %
unnecessary edit rate
```

---

# 16. Stage 3 — repository SFT

## Objective

Move from one-file coding to software engineering.

Dataset:

```text
SWE-smith
custom repository tasks
selected CommitPackFT
```

Context:

```text
16K–32K
```

Teach explicitly:

```text
issue → locate files → inspect dependencies → patch → tests
```

Do not give every example the exact files that must be edited.

A portion of training examples should require **file selection**.

Suggested mixture:

```text
30% file localization
15% function localization
20% bug explanation
25% patch generation
10% test generation
```

---

# 17. Stage 4 — 32K–64K repository context

At this stage, generic coding examples should become a minority.

Example distribution:

```text
repository engineering     70%
editing/debugging           20%
generic coding rehearsal    10%
```

Context curriculum:

```text
32K   60%
48K   20%
64K   20%
```

The purpose of the 10% generic rehearsal set is to reduce catastrophic specialization.

---

# 18. Stage 5 — 64K–128K specialization

This stage should be **small but high quality**.

Do not spend huge amounts of compute repeating short coding examples at 128K.

Recommended examples:

```text
2K–10K high-quality long-context tasks
```

depending on how many can be verified.

Context distribution:

```text
64K       40%
80–96K    30%
96–128K   30%
```

A more conservative first run:

```text
64K       60%
96K       30%
128K      10%
```

Then increase 128K only if evaluation shows improvement.

## Long-context anti-shortcut rules

Randomize:

- relevant-file position
- repo tree position
- issue placement
- number of distractor files
- target file count
- target function count

Include examples where:

```text
the correct action is NO CHANGE
```

or:

```text
insufficient information
```

to reduce hallucinated patches.

---

# 19. Packing strategy

For short-stage SFT, use packing when supported.

Example:

```text
example A 1.2K
example B 2.5K
example C 900
example D 3K
----------------
packed ≈ 7.6K
```

This improves utilization.

For long repository examples, avoid mixing unrelated repository tasks into a single attention context unless the trainer's packing implementation properly isolates sequence attention/loss behavior.

Treat each large repository problem as one semantic sequence.

---

# 20. Initial Unsloth training skeleton

The exact Qwen3.5 support may evolve, so pin tested package versions.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MAX_SEQ_LENGTH = 8192

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-2B",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="outputs/qwen35-2b-coder-stage1",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=1,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=250,
        eval_steps=250,
        bf16=True,
        packing=True,
        report_to="tensorboard",
    ),
)

trainer.train()
```

**Important:** Verify current API names in the installed Unsloth/TRL versions before a production run.

---

# 21. LoRA target modules

Do not blindly copy target-module names from Qwen2.5 or Llama.

Qwen3.5 uses a hybrid architecture, so first inspect the actual model:

```python
for name, module in model.named_modules():
    if "Linear" in module.__class__.__name__:
        print(name, module.__class__.__name__)
```

Then decide whether to adapt:

- attention projections
- linear-attention projections
- MLP projections
- output projections

Run an ablation:

```text
A. attention projections only
B. attention + MLP
C. all supported linear modules
```

Compare quality and adapter size.

For v1, favor **broad but conventional LoRA coverage** over trying to modify embeddings or LM head.

---

# 22. MTP handling

Qwen3.5 exposes MTP capability and serving stacks can use it for speculative decoding.

For the first coder SFT:

> Treat MTP primarily as an inference/deployment feature.

Do not assume a normal SFT run is optimizing a separate MTP auxiliary objective unless the exact Unsloth/Qwen recipe explicitly supports it.

After merging:

```text
normal decoding benchmark
vs
MTP speculative decoding
```

Measure:

```text
tokens/sec
latency
output equality / quality
termination behavior
```

Quality should remain the primary criterion.

---

# 23. Training hardware

## Development

For Stage 1 experiments:

```text
A100 40GB Colab     batch 1 × accumulation 16 at 8K
A100/H100 80GB      batch 2 × accumulation 8 at 8K
```

The implementation requires native BF16 support and rejects smaller or non-BF16 GPUs for the validated profiles.

## Long-context stages

For serious 64K–128K runs:

```text
A100 80GB
H100 80GB
H100 NVL 94GB
H200 141GB
```

are safer.

Unsloth has published long-context memory results demonstrating that its gradient-checkpointing approach can dramatically extend trainable sequence length, but those published numbers are architecture- and setup-specific. **Benchmark Qwen3.5-2B itself before assuming a particular maximum length.**

---

# 24. Memory experiment before real 128K training

Run this progression with batch size 1:

```text
8K
16K
32K
48K
64K
80K
96K
112K
128K
```

At each point record:

```text
peak VRAM
host RAM
tokens/sec
step time
GPU utilization
checkpoint overhead
OOM / stability
```

Create:

```text
memory_profile.csv
```

Stop guessing about GPU requirements after this experiment.

---

# 25. Compute-cost methodology

Cloud GPU prices change frequently.

As a September 2026 reference, RunPod currently lists approximately:

```text
A100 80GB PCIe   ~$1.19/hr
H100 PCIe 80GB   ~$1.99/hr community-style listing
H100 SXM 80GB    ~$2.69/hr
H200 141GB       ~$3.59/hr
```

On-demand / secure offerings can be higher; for example current H100 on-demand references are around the high-$2/hour range.

Do not estimate the project from wall-clock guesses alone.

After the pilot, compute:

```text
cost =
    GPU hourly price
    ×
    measured GPU hours
    ×
    number of runs
```

## Practical project budget

Reserve budget for:

```text
1 baseline
3 LR/LoRA experiments
1 Stage-1 full run
1 Stage-2 run
1 Stage-3 run
2 long-context experiments
1 final run
```

A reasonable exploratory budget is:

```text
~$100–$300
```

for BF16 LoRA experimentation on marketplace/cloud GPUs.

The actual final run may cost much less than the total budget; experiments, failures and ablations consume most of the budget.

---

# 26. Evaluation harness

Build evaluation **before** large training.

Suggested repository:

```text
eval/
├── humaneval_eval.py
├── mbpp_eval.py
├── livecodebench_eval.py
├── file_localization.py
├── patch_eval.py
├── test_runner.py
├── long_context_eval.py
└── report.py
```

Every checkpoint should produce:

```text
reports/<checkpoint>/
├── summary.json
├── coding.csv
├── editing.csv
├── repo.csv
├── long_context.csv
└── generations.jsonl
```

---

# 27. Internal benchmark set

Create at least 200–500 private tasks not used for training.

Suggested distribution:

| Type | Count |
|---|---:|
| Generate function/module | 50 |
| Debug | 50 |
| Edit existing code | 75 |
| Add tests | 30 |
| Multi-file change | 50 |
| Repository localization | 50 |
| Long-context tasks | 50 |

Use Python heavily, but include the actual target languages you care about.

If the intended coder is general purpose, a useful initial mix is:

```text
Python       35%
JavaScript   15%
TypeScript   15%
Java         10%
C++          10%
Go            8%
Rust          5%
Shell/SQL     2%
```

Adjust for intended deployment.

---

# 28. Long-context evaluation design

Create controlled tests where the only source of truth is inside the supplied context.

Example:

```text
tokens 0–20K        unrelated files
tokens 20–50K       related package
tokens 50–90K       distractor implementation
tokens 90–110K      true implementation
tokens 110–125K     failing test
tokens 125–128K     issue
```

Ask:

```text
Which file is wrong?
What is the root cause?
Produce the patch.
```

Rotate positions across examples.

Metrics:

```text
localization accuracy
exact bug identification
patch test pass
hallucinated dependency rate
```

A model that merely remembers the first 32K will fail visibly.

---

# 29. Training gates

Do not move automatically from one stage to the next.

## Stage 1 gate

```text
core coding improved
general instruction quality retained
```

## Stage 2 gate

```text
patch correctness improved
no major code-generation regression
```

## Stage 3 gate

```text
file localization improved
multi-file tasks improved
```

## Stage 4 gate

```text
32K/64K performance improved
8K coding not materially degraded
```

## Stage 5 gate

```text
96K/128K performance improved
64K not degraded
core coding retained
```

If a stage fails:

```text
reduce LR
increase rehearsal data
reduce long-context proportion
rollback checkpoint
```

---

# 30. Ablations worth running

At minimum:

## A. Base vs coder SFT

```text
Qwen3.5-2B
vs
Stage-1 coder
```

## B. Generic coding vs edit data

```text
OpenCodeInstruct only
vs
OpenCodeInstruct + CommitPackFT
```

## C. Repo supervision

```text
no SWE-smith
vs
+ SWE-smith
```

## D. Long-context training

```text
max 32K
vs
max 64K
vs
max 128K
```

## E. Long-context proportion

```text
5%
10%
20%
```

## F. LoRA rank

```text
r=16
r=32
r=64
```

## G. Training target

```text
attention only
vs
attention + MLP
```

These ablations tell us where the gains actually come from.

---

# 31. Logging

Use local TensorBoard logs during the Colab run. No W&B credential is required.

Log:

```text
train loss
eval loss
learning rate
gradient norm
tokens/sec
samples/sec
GPU memory
CPU memory
sequence length
mean packed utilization
checkpoint size
```

Also log evaluation metrics by context length.

Do not report only one aggregate number.

---

# 32. Checkpoint naming

Use explicit lineage.

Example:

```text
q35-2b-coder-s0-base
q35-2b-coder-s1-oci-r32
q35-2b-coder-s2-edit
q35-2b-coder-s3-repo32k
q35-2b-coder-s4-repo64k
q35-2b-coder-s5-repo128k
q35-2b-coder-final
```

This makes ablations reproducible.

---

# 33. Recommended repository structure

```text
Qwen3.5-2B-Coder/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── stage1_8k.yaml
│   ├── stage2_16k.yaml
│   ├── stage3_32k.yaml
│   ├── stage4_64k.yaml
│   └── stage5_128k.yaml
│
├── data/
│   ├── raw/
│   ├── filtered/
│   ├── repo_tasks/
│   └── manifests/
│
├── scripts/
│   ├── download_data.py
│   ├── load_eer6_opencode.py
│   ├── stratify_broad.py
│   ├── deduplicate.py
│   ├── build_commit_tasks.py
│   ├── build_repo_tasks.py
│   ├── tokenize_stats.py
│   ├── train_sft.py
│   ├── merge_lora.py
│   └── export_gguf.sh
│
├── eval/
│   ├── coding/
│   ├── editing/
│   ├── repo/
│   └── long_context/
│
├── outputs/
└── reports/
```

---

# 34. Dataset manifest

Every generated dataset should have a manifest:

```yaml
name: coder_stage3_repo
version: 1
created_at: 2026-09-04

sources:
  - SWE-smith-py
  - CommitPackFT
  - internal_repo_tasks

examples: 42000

token_stats:
  p50: 8200
  p90: 24100
  p99: 31800

languages:
  python: 0.55
  javascript: 0.10
  typescript: 0.10
  java: 0.10
  cpp: 0.05
  go: 0.05
  rust: 0.05

dedup:
  method: minhash
  benchmark_exclusion: true
```

This is essential for repeatability.

---

# 35. Avoid these mistakes

## Mistake 1 — Train 5M OpenCodeInstruct rows immediately

Result:

```text
high cost
duplicate behavior
too much synthetic style
little repo specialization
```

## Mistake 2 — Set max length to 128K from day one

Result:

```text
slow iterations
low sample throughput
wasted tokens
hard debugging
```

## Mistake 3 — Measure only HumanEval

You could improve toy code generation while making repository editing worse.

## Mistake 4 — Leak repositories across splits

This makes repo-level evaluation misleading.

## Mistake 5 — Put relevant file first every time

The model learns position bias instead of long-context search.

## Mistake 6 — Train huge reasoning transcripts

For a 2B model, concise high-quality engineering traces are generally more useful than bloated synthetic reasoning.

## Mistake 7 — Export GGUF before evaluating merged Safetensors

First establish:

```text
adapter quality
→ merged model quality
→ quantized model quality
```

Otherwise quantization and SFT effects become confounded.

---

# 36. Final deployment experiment

Evaluate:

```text
BF16
Q8_0
Q6_K
Q5_K_M
Q4_K_M
```

Measure:

```text
HumanEval/MBPP delta
internal patch score
repo localization
tokens/sec
RAM
startup time
128K KV-cache memory
```

Likely deployment target:

```text
Q5_K_M
```

if quality matters more than absolute size, and:

```text
Q4_K_M
```

if the goal is smallest practical local inference.

Use `llama.cpp` for GGUF deployment and compare MTP/speculative decoding where supported.

---

# 37. Minimal v1 experiment

If the goal is to validate the idea quickly, do this first.

## Dataset

```text
35K EER6 OpenCodeInstruct Refined
15K EER6 OpenCodeInstruct Broad
10K CommitPackFT / debug-edit examples
5K SWE-smith / custom repo tasks
```

## Training

```text
Stage A: 8K
Stage B: 32K
```

Skip 128K initially.

## Evaluation

```text
HumanEval+
MBPP+
100 internal edit tasks
50 repository tasks
32K code-needle benchmark
```

If this beats baseline:

```text
scale dataset
→ 64K
→ 128K
```

This is much safer than immediately launching a large training job.

---

# 38. Preferred full experiment

After v1 validates the pipeline:

```text
                  Qwen3.5-2B
                       │
                       ▼
              Baseline evaluation
                       │
                       ▼
          ~420K core coding SFT
       250K Refined + 150K Broad
          + ~20K debug/special
                    8K
                       │
                       ▼
           50K–100K editing/debug
                  8K–16K
                       │
                       ▼
             20K–40K repo tasks
                  16K–32K
                       │
                       ▼
             10K–20K long repo
                  32K–64K
                       │
                       ▼
              2K–10K premium
                 64K–128K
                       │
                       ▼
               Final evaluation
                       │
                       ▼
                 Merge LoRA
                       │
                       ▼
              GGUF Q5/Q4 + MTP
                       │
                       ▼
                  llama.cpp
```

---

# 39. Definition of done

Call the project successful only if the final model demonstrates all of the following:

- [ ] better coding pass rate than base Qwen3.5-2B
- [ ] better debugging performance
- [ ] better code-edit performance
- [ ] better file localization
- [ ] better multi-file patch success
- [ ] useful 64K behavior
- [ ] measurable 128K utilization
- [ ] no severe general instruction regression
- [ ] no severe repetition/termination regression
- [ ] merged Safetensors reproduces adapter performance
- [ ] GGUF retains acceptable quality
- [ ] llama.cpp runs at target context
- [ ] benchmark results and dataset manifests are reproducible

---

# 40. Suggested execution order

## Week / Sprint 1 — Pipeline

```text
Day 1
- baseline model
- inference harness
- benchmark harness

Day 2
- download OpenCodeInstruct
- filtering
- deduplication
- token statistics

Day 3
- 5K-example smoke SFT
- verify loss
- verify merge
- verify inference

Day 4
- 50K Stage-1 SFT
- evaluate

Day 5
- editing dataset
- Stage-2 SFT
- evaluate
```

## Sprint 2 — Repository behavior

```text
- prepare SWE-smith
- build file-localization tasks
- build patch tasks
- train 16K/32K checkpoint
- evaluate repo behavior
```

## Sprint 3 — Long context

```text
- create 32K/64K/128K repo examples
- profile GPU memory
- train 64K
- gate
- train 128K
- gate
```

## Sprint 4 — Finalization

```text
- ablations
- final merge
- GGUF export
- quantization comparison
- llama.cpp benchmark
- release model card
```

---

# 41. Final recommendation

For the first high-quality model, optimize for:

```text
Qwen3.5-2B
+
EER6 OpenCodeInstruct Refined as quality anchor
+
EER6 OpenCodeInstruct Broad as diversity pool
+
real code edits
+
repository localization
+
verified patches
+
small amount of carefully constructed 128K data
```

not:

```text
Qwen3.5-2B
+
millions of generic coding prompts
+
everything padded/packed to 128K
```

The research hypothesis should be:

> A native-long-context 2B model can become a useful coding specialist if SFT focuses on executable correctness, code editing, repository localization and sparse long-context evidence rather than only short-form code generation.

That is a much stronger target than simply creating another “2B model fine-tuned on coding instructions.”

---

# 42. Source references

1. Qwen3.5-2B / Unsloth model repository  
   https://huggingface.co/unsloth/Qwen3.5-2B

2. Official Qwen3.5-2B configuration  
   https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json

3. EER6 OpenCodeInstruct Broad  
   https://huggingface.co/datasets/EER6/nvidia-OpenCodeInstruct-broad

4. EER6 OpenCodeInstruct Refined  
   https://huggingface.co/datasets/EER6/nvidia-OpenCodeInstruct-refined

5. NVIDIA OpenCodeInstruct  
   https://huggingface.co/datasets/nvidia/OpenCodeInstruct

6. SWE-smith Python  
   https://huggingface.co/datasets/SWE-bench/SWE-smith-py

7. SWE-smith collection  
   https://huggingface.co/collections/SWE-bench/swe-smith

8. CommitPackFT  
   https://huggingface.co/datasets/bigcode/commitpackft

9. TRL SFTTrainer  
   https://huggingface.co/docs/trl/sft_trainer

10. Unsloth long-context gradient checkpointing  
   https://unsloth.ai/blog/long-context

11. Unsloth fine-tuning notebook collection  
   https://github.com/unslothai/notebooks

12. RunPod GPU catalogue / current indicative pricing  
    https://www.runpod.io/gpu-models

---

## Immediate next implementation milestone

Build these four scripts first:

```text
01_prepare_eer6_opencode.py
02_stratify_broad.py
03_build_coder_dataset.py
04_train_qwen35_coder.py
05_eval_coder.py
```

The first end-to-end target should be:

```text
Qwen3.5-2B
→ 35K EER6 Refined
→ 15K EER6 Broad
→ optional 10K edit/debug examples
→ BF16 LoRA
→ 8K context
→ HumanEval+/MBPP+/editing evaluation
```

Only after this pipeline is stable should the project expand to 32K, 64K and 128K repository SFT.
