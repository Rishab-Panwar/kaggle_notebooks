# NPPE-2: Multilingual Speech Recognition - Case Study

> **Competition:** NPPE-2 Multilingual Speech Recognition · Kaggle Community Competition  
> **Task:** Transcribe audio recordings across English, Hindi, and Tamil  
> **Metric:** Word Error Rate (WER) - lower is better  
> **Public Score:** 0.19228 | **Private Score:** 0.19228 | Ranked **13/190** teams
> **Competition link:** https://www.kaggle.com/t/3a21e1175d3048a686531cc96c197857 
---

## The Problem

The task was to build a system that takes raw `.wav` files and outputs verbatim transcripts - across three languages with very different phonetics: English, Hindi, and Tamil. That sounds straightforward. It wasn't.

The real challenge wasn't transcription. It was **language identification**. To get a good transcript, you need to pick the right model for each audio clip. To pick the right model, you need to know the language first. And if the language detector gets it wrong - which it does, a lot with Indic audio - everything downstream falls apart.

The secondary challenge was **compute**. Kaggle Notebooks impose strict GPU memory limits. Every model choice had to fit within budget.

---

## How I Thought About It

My first instinct was to use a single model (Whisper Large-v3) for everything - it supports 99 languages and is state-of-the-art. But after running it on a sample of the training data, it was clear that Whisper's Hindi and Tamil transcription quality is noticeably weaker than its English performance. This makes sense: Whisper's training data skews heavily toward English and European languages.

So the real question became: **can I build a hybrid system that uses a specialized Indic model for Hindi/Tamil and Whisper only for English?**

The answer was yes - but only if the routing between models was reliable. That's where I spent most of my time.

The pipeline I landed on:

```
audio → Language ID (Whisper-base) → routing table → transcription model → normalize → submission
```

---

## How I Architected It

### The Dual-Model Strategy

Two models run simultaneously on two GPUs:

| Model | GPU | Used For |
|---|---|---|
| AI4Bharat IndicConformer-600M-multilingual | cuda:0 | Hindi and Tamil transcription |
| OpenAI Whisper Large-v3 | cuda:1 | English transcription (and fallback) |
| OpenAI Whisper-base | cuda:0 (shared) | Language identification only |

Keeping both models loaded at all times (instead of loading/unloading per sample) was critical for inference speed across ~2,000 test samples.

### Two Audio Loaders, Not One

Whisper and IndicConformer expect different input formats:

- **Whisper** takes a float32 numpy array → internally converts to log-mel spectrogram
- **IndicConformer** takes a `(1, T)` float32 PyTorch tensor → operates directly on the waveform

Both loaders share the same preprocessing: convert to mono and resample to 16kHz. This matters because raw competition audio came in at different sample rates (44.1kHz and others), but both models were pretrained at 16kHz.

### The Routing Table - The Core of the Solution

After running Whisper-base language detection on the full training set, I did a careful error analysis. The findings shaped the entire architecture:

**Whisper misdetects Indic audio at an alarming rate.**

The two biggest problems:
- **65 Hindi audio clips** were misdetected as `ur` (Urdu) - the highest single source of routing error
- **41 Tamil audio clips** were misdetected as `ml` (Malayalam)
- Tamil was also frequently misdetected as `ja` (Japanese), `kn` (Kannada), `te` (Telugu), `tl` (Filipino)
- Hindi was misdetected as over 20 different language codes including `ro`, `pt`, `cs`, `tr`, `ru`, and `ar`

The fix was not to trust the raw LID output. Instead, I built a **routing table** that collapses ~30 possible LID outputs into three routes:

```python
TAMIL_MISDETECTIONS = {"ta", "ml", "kn", "te", "ja", "tl"}
HINDI_MISDETECTIONS = {"hi", "ur", "ne", "mr", "pa", "bn", "ro", "pt", "cs", "tr", ...}
# Everything else → Whisper (English)
```

If Whisper-base says `ur`, the audio goes to IndicConformer in Hindi mode - not blindly trusted as Urdu. If it says `ml`, the audio goes to IndicConformer in Tamil mode. This one routing decision fixed the majority of Indic transcription errors.

---

## Specific Technical Decisions and Tradeoffs

**RNNT over CTC for IndicConformer.** IndicConformer supports two decoding modes. RNNT gave noticeably better WER on Indic transcriptions during validation. CTC was faster, but the accuracy gap was large enough that RNNT was worth the extra compute.

**Whisper-base for LID, not Whisper Large-v3.** Language detection is a much simpler task than transcription - you only need an ISO 639-1 code, not a full transcript. Running LID on the large model would have wasted GPU memory and time. Whisper-base is fast and gives accurate enough language probs to build a routing table from.

**float16 for Whisper Large-v3.** Loading in half precision cuts the VRAM footprint roughly in half (from ~6GB to ~3GB) with negligible WER impact on a GPU with Tensor Cores. This was the difference between fitting on a single GPU and not fitting at all.

**Beam search for English, RNNT for Indic.** Whisper with `num_beams=5` outperformed greedy decoding on longer English clips. For Indic, beam search is baked into RNNT's architecture.

**Whisper-base as an empty-prediction fallback.** If IndicConformer returned an empty string on any test clip (which happened occasionally on very short or degraded audio), that sample was retried with Whisper Large-v3. If even that failed, a placeholder word (`"the"`) was inserted. A blank row is treated as a full deletion by the WER scorer - much worse than a single wrong word.

**Text normalization: lowercase + whitespace collapse only.** The competition organizer confirmed lowercase is accepted for English. No punctuation stripping, no more complex cleaning - overengineering normalization can hurt WER if the ground truth contains punctuation.

---

## What I Tried That Didn't Work

**Using a single Whisper model for all three languages.** Whisper Large-v3 handles English very well. For Tamil it was mediocre. For Hindi it was inconsistent - sometimes good, sometimes producing transliterated English-script output instead of Devanagari. A single-model approach was a ceiling I couldn't get past.

**Wav2Vec2.** Ruled out early. Its pretraining is self-supervised, so it needs substantial labeled fine-tuning to be useful for transcription. With only ~2,000 labeled training samples and strict compute limits, fine-tuning Wav2Vec2 wasn't viable within this competition's constraints.

**Whisper-medium as the Indic model.** Tried this before finding IndicConformer. Medium gave slightly better Hindi than base, but still produced poor Tamil. The jump from any Whisper variant to IndicConformer (pretrained on 22 Indian languages) was much more significant than going from Whisper-medium to Whisper-large for Indic transcription.

**Trusting raw Whisper LID output without a routing table.** The first version of the pipeline used the raw detected language to pick a model directly. It routed ~65 Hindi samples to Urdu and ~41 Tamil samples to Malayalam. WER on those samples was extremely high. Building the routing table was the single highest-impact change I made.

---

## Validation Methodology

Before running on test data, I ran the full pipeline on 100 training samples and computed:
- Per-sample WER (to surface specific failing clips)
- WER broken down by **route** (to catch which model was underperforming)
- WER broken down by **raw detected language** (to catch which misdetections were most costly)

The "wrong predictions" table sorted by WER descending made it easy to inspect the worst failures manually and decide whether the issue was routing, normalization, or the model itself.

---

## Stack

| Component | Library/Model |
|---|---|
| Indic ASR | `ai4bharat/indic-conformer-600m-multilingual` |
| English ASR | `openai/whisper-large-v3` |
| Language ID | `openai-whisper` (base model) |
| Audio I/O | `librosa`, `torchaudio`, `soundfile` |
| WER evaluation | `jiwer` |
| Training data EDA | `langdetect`, `matplotlib`, `seaborn` |
| Inference | Kaggle Notebooks, 2× GPU (T4/P100) |

---

## Key Takeaway

The highest-leverage insight in this project was not model selection - it was understanding that Whisper's language detector is systematically unreliable on Indic audio and building a routing layer that corrects for its specific failure modes. The routing table, built from a careful read of ~2,000 training-set LID errors, turned a mediocre single-model pipeline into a robust hybrid one.
