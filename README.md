# NPPE-1: Multilingual Sentiment Analysis - Case Study

> **Competition:** NPPE-1 Multilingual Sentiment Analysis - Kaggle Community Competition  
> **Task:** Classify sentiment (Positive / Negative) across 13 Indian languages  
> **Metric:** Macro F1 Score  
> **Model:** Gemma 3-1B-IT fine-tuned with QLoRA

---

## The Problem

The task was to classify sentiment in sentences across 13 Indian languages: Assamese, Bodo, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and Urdu. Binary classification - Positive or Negative.

On the surface, that looks simple. The dataset made it genuinely hard.

**900 training samples across 13 languages.** That averages to roughly 70 examples per language. Some languages had fewer. For context, that is not enough data to meaningfully fine-tune a model from scratch in any single language, let alone generalize across 13 at once.

**The rules added another constraint:** only LLaMA 3.1-8B-Instruct or Gemma 3-1B-IT were allowed. No external models, no external data, Kaggle Notebooks only. So the entire solution had to fit within a T4 GPU's memory budget.

The core question was not "which model do I use?" It was "how do I make a 1B parameter model reliably classify sentiment across 13 scripts, 13 language families, and wildly different amounts of training signal, inside a 9-hour session on a T4?"

---

## How I Thought About It

The first thing I checked was whether the model could do this task at all without fine-tuning. Gemma 3-1B-IT is an instruction-tuned model - it can already follow natural language instructions. If the zero-shot baseline is reasonable, fine-tuning becomes an optimization problem. If it is broken, fine-tuning becomes a rescue mission.

So I ran the vanilla model on a stratified 200-sample subset before touching the training data. That number became the baseline to beat.

The second thing I thought about was what fine-tuning actually needed to teach the model. It already knows what "Positive" and "Negative" mean. It has seen Hindi during pretraining. The job of the LoRA adapters was narrower: reinforce the output format across all 13 scripts, fix the languages where the base model is weakest, and ensure the model does not drift into generating long explanations instead of a single word.

That framing shaped every decision: prompt design, LoRA config, training setup, and inference strategy.

---

## How I Architected It

### The Stack

- **Base model:** Gemma 3-1B-IT, loaded in 4-bit NF4 quantization via bitsandbytes
- **Adapter:** LoRA (r=32) attached to all linear layers via PEFT
- **Training:** SFTTrainer from TRL with completion-only loss masking
- **Inference:** 3-pass majority vote (1 greedy + 2 low-temperature samples)

### Prompt Design

Every prompt includes four hand-picked few-shot examples covering Tamil, Bengali, Urdu, and Telugu - deliberately chosen to span different scripts. The model sees concrete anchors before classifying the target sentence. Without few-shot examples, the model was inconsistent about output format across scripts it had less exposure to during pretraining.

The prompt ends with an explicit constraint: "Reply with exactly one word: Positive or Negative." This sounds obvious but it matters. Without it, the model occasionally generates a sentence explaining its reasoning instead of a label.

The same `make_prompt` function handles both training and inference. At training time it appends the ground truth label. At inference time it stops right after the response template and lets the model complete. Using the same function eliminates any mismatch between what the model was trained on and what it sees during inference.

### Completion-Only Loss Masking

This was the most important training decision. SFTTrainer was configured with `completion_only_loss=True`, which means the cross-entropy loss is computed only on the model's response tokens - "Positive" or "Negative" - not on the prompt or few-shot examples.

Without this, the model receives gradient signal for predicting every token in the prompt, including the few-shot examples and the instruction text. That dilutes the training signal. With completion-only masking, every gradient step is directly about learning to predict the correct label.

### Using All 900 Samples for Training

There was no train/validation split. With 70 samples per language on average, holding out 20% would have left some languages with fewer than 15 training examples. That is not a validation set; it is just noise that starves the model of the signal it needs.

Instead, the baseline evaluation (pre-fine-tuning) was run on a stratified 200-sample subset before training began. The post-fine-tuning evaluation was run on the exact same subset after loading the adapter. This apples-to-apples comparison is what justified the fine-tuning effort, without needing to hold out data from training.

---

## Specific Technical Decisions and Tradeoffs

**NF4 4-bit quantization with double quantization.** NormalFloat-4 is specifically designed for normally distributed model weights and gives better accuracy than standard int4 at the same memory footprint. Double quantization (also quantizing the quantization constants) saves roughly another 0.4 bits per parameter. Together, a 1B model sits at around 1.5 GB of VRAM, leaving substantial headroom for activations and optimizer states on a T4.

**LoRA rank r=32, alpha=64, targeting all linear layers.** The common default is r=16 on attention layers only. I went to r=32 and added MLP layers (gate_proj, up_proj, down_proj) because cross-lingual sentiment adaptation is a harder task than domain adaptation within one language. More expressiveness in the adapters was worth the slightly higher memory cost on a 1B model, where even r=32 adds only a small fraction of trainable parameters.

**Effective batch size of 64 via gradient accumulation.** T4 VRAM limits per-device batch size to 8 with this setup. Accumulating gradients over 8 steps before updating gives an effective batch of 64, which stabilizes training on a 900-sample dataset where individual batches would otherwise be noisy.

**8 training epochs.** More than typical. With 900 samples, each epoch is short. 8 epochs gave the model enough passes to internalize the pattern across all 13 scripts without visible overfitting (monitored by comparing pre/post F1 on the fixed baseline split).

**paged_adamw_8bit optimizer.** AdamW optimizer states are normally stored in fp32, which for a 1B model with LoRA adapters would be manageable but still a meaningful chunk of T4 VRAM. 8-bit Adam stores optimizer states in 8-bit precision with paged memory management. This freed up headroom for longer sequences and larger batches.

**3-pass majority vote at inference.** A single greedy decoding pass is fast but can be overconfident on ambiguous inputs. Two additional passes at temperature 0.3 introduce just enough stochasticity to break ties when the model is uncertain. The majority vote across three runs consistently outperformed single-pass greedy on the baseline sample. The cost is 3x inference time, which was acceptable for 100 test samples.

**Left padding during generation, right padding during training.** Causal LMs generate by continuing from the rightmost token. During generation, padding must be on the left so all prompts in a batch end at the same position. During training, padding is on the right so the loss is computed correctly on the response tokens. Swapping `tokenizer.padding_side` between the two phases is a small detail that silently breaks things if missed.

---

## What I Tried That Didn't Work

**LLaMA 3.1-8B-IT.** The 8B model is the stronger model in principle, but it would not fit on a T4 alongside the LoRA training overhead. With 4-bit quantization the base model alone took around 5-6 GB, and the peak VRAM during training pushed into OOM territory. Reducing batch size to compensate made training too unstable on 900 samples. Gemma 3-1B-IT was the practical choice given the hardware constraint.

**LoRA on attention layers only.** The first adapter configuration targeted only `q_proj`, `k_proj`, `v_proj`, `o_proj`. Post-FT F1 improvement was modest. Adding the MLP layers (gate_proj, up_proj, down_proj) gave a meaningful bump - the model was apparently not only adjusting attention patterns but also updating its internal representations for Indic scripts.

**Single-pass greedy inference.** Clean, fast, and usually right for languages with stronger training signal (Hindi, Tamil, Bengali). For lower-resource languages in the dataset (Bodo, Assamese, Odia), the greedy pass was noticeably more error-prone. The 3-pass majority vote closed most of that gap.

**Training with loss on full prompt.** An early version used standard SFT loss over the entire sequence - prompt and completion. Per-class F1 was lower, particularly for the minority class, because the gradient signal was spread across hundreds of prompt tokens and the model was effectively trying to memorize the few-shot examples rather than learn the classification signal.

---

## Validation Methodology

The baseline evaluation ran before any fine-tuning, on a stratified 200-sample split drawn from the training set. The same 200-sample split was used for post-FT evaluation without resampling. The only variable that changed between the two evaluations was the model weights. The delta in Macro-F1 between the two runs is the cleanest measure of what the LoRA adapters actually contributed.

Per-language accuracy was tracked at baseline to identify which of the 13 languages the vanilla model handled worst. Those became the focus of qualitative prompt iteration before finalizing the few-shot examples.

---

## Stack

| Component | Library / Tool |
|---|---|
| Base model | `google/gemma-3-1b-it` |
| Quantization | `bitsandbytes` (NF4 4-bit, double quant) |
| LoRA adapters | `peft` (r=32, all linear layers) |
| Training | `trl.SFTTrainer` with completion-only loss |
| Optimizer | `paged_adamw_8bit` |
| Evaluation | `sklearn.metrics.f1_score` (Macro-F1) |
| Hardware | Kaggle Notebooks, single T4 GPU |

---

## Key Takeaway

The highest-leverage decision in this project was not model selection or hyperparameter tuning. It was recognizing that with 70 samples per language, the training signal is too sparse to teach a model anything new about language structure. What fine-tuning could realistically accomplish was much narrower: enforce a consistent output format, reinforce the correct label distribution per script, and fix the specific failure modes visible in the baseline evaluation. Scoping the problem to that smaller goal shaped the prompt design, the loss masking strategy, and the decision to use all 900 samples for training rather than holding out a validation set.
