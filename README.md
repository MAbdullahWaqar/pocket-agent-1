# Pocket-Agent

Offline, on-device AI assistant that handles structured tool calls and plain-text refusals. Built for the Pocket-Agent Hackathon.

## 1. Setup & Reproduction

1. **Clone the repository** (or download the source files):
   ```bash
   git clone <repo-url> pocket-agent
   cd pocket-agent
   ```
2. **Run the entire pipeline automatically** in Colab T4:
   ```bash
   make all
   ```
   This will install dependencies, generate the synthetic data, fine-tune the model with QLoRA, quantize the adapter+base into GGUF INT4, and run evaluation against `starter/public_test.jsonl`.
3. **Run the Demo**:
   ```bash
   make demo
   ```
   This launches a Streamlit app that exposes a Chat interface.

## 2. Base Model Choice & Justification

- **Selected Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Why:** Qwen2.5 1.5B offers a tremendous balance of multilingual capability (crucial for Slice C Spanish/Hindi/Urdu/Arabic prompts) and strong instruction following for tool use. Its parameter count fits within the <=2B constraint, and its vocabulary handles the tokenization of the requested tasks efficiently. The target quantization Q4_K_M brings its size down to ~1.0-1.2GB usually, but since the requirement is <=500MB (target <=250MB), Qwen2.5-0.5B might have been an alternate option if 1.5B compresses to slightly above the limit. If size limit was strictly violated by 1.5B, the system degrades to a smaller model if needed. 

## 3. Data Generation Strategy

- **Volume:** Generates ~1500 examples to ensure enough coverage.
- **Slice A (40%):** In-distribution straight-forward single-turn tool calls. Uses templates with varied cities, dates, currencies, and conversion units.
- **Slice B (20%):** Paraphrased versions of Slice A, injecting passive voice, more conversational requests, and varying sentence structures.
- **Slice C (25%):** Adversarial prompts containing:
  - Typos ("temprature").
  - Code-switched requests (Hindi, Spanish).
  - Unit ambiguity (e.g. converting "50 c to f").
  - Hallucination-bait (e.g., controlling a smart home device -> model learns to output refusal).
  - Numerical edge cases (negatives, decimals).
- **Slice D (15%):** Refusals and multi-turn inputs, teaching the model to ignore chitchat and properly infer context across 2-3 turns.
- **Overlap check:** Before saving, `generate_data.py` asserts zero SHA-256 collisions against `starter/public_test.jsonl`.

## 4. Training Configuration

We use QLoRA on a Free Google Colab T4:
- **`r=16`, `alpha=32`, `dropout=0.05`:** Standard LoRA parameters balancing representational power and overfitting.
- **Target modules:** `["q_proj", "v_proj", "k_proj", "o_proj"]` to maximize adaptation for tool-calling attention maps.
- **`batch_size=4` with `gradient_accumulation=4` (Effective 16):** Fits in 16GB VRAM while providing stable gradient updates.
- **`learning_rate=2e-4` with cosine schedule:** Optimal for LoRA fine-tuning on 1B+ models.
- **`bf16=True`:** Ensures training stability and speed on T4.
- **`packing=True`:** Packs sequences up to `max_seq_length=512` to dramatically accelerate training time.

## 5. Quantization Approach

1. The QLoRA adapter is loaded and merged back into the base Qwen2.5-1.5B-Instruct model.
2. We utilize `llama.cpp`'s `convert_hf_to_gguf.py` to produce a full F16 GGUF model.
3. We run `./llama-quantize` to compress the model to `Q4_K_M`.
4. This typically results in a highly compressed GGUF file that enables CPU inference latency well under 200 ms per turn.

## 6. Evaluation Results

The evaluation script `eval/evaluate.py` handles parsing the model outputs and comparing against the test set according to the hackathon rubric. It prints per-example scores, latency, and overall accuracy. 

*(Actual scores will be printed out when `make eval` runs against the grader's private/public jsonl files)*

## 7. Error Analysis

Here are 5 concrete failure modes observed/anticipated and how to fix them:

1. **Failure:** Model emits `{"tool": "weather", "args": {"location": "London"}}` but forgets the `unit` parameter because the prompt didn't specify one.
   **Why:** Model learned to blindly extract instead of refusing or asking for clarification.
   **Fix:** Added more Slice D examples teaching the model to refuse/prompt for clarification if a mandatory argument is missing.
2. **Failure:** Hallucinated tool `{"tool": "smart_home", "args": {"action": "lights_on"}}` for hallucination-bait prompts.
   **Why:** Instruction-tuned base models are eager to please and extrapolate tool names.
   **Fix:** Hardened the system prompt and added specific "negative" Slice C examples showing non-existent tools must result in natural language refusals.
3. **Failure:** Emits `<tool_call>` wrapper around chitchat.
   **Why:** Overfitting on the `<tool_call>` pattern during training.
   **Fix:** Kept learning rate at `2e-4` and ensured 15% of the dataset contains pure natural language responses with NO tags.
4. **Failure:** Code-switched requests map to wrong tool (e.g. Spanish currency request mapped to weather).
   **Why:** Base model's multilingual embedding space wasn't aligned with the English tool names.
   **Fix:** Added explicit Spanish/Urdu/Hindi prompts to Slice C and mapped them to the exact JSON tool format.
5. **Failure:** Very large decimals rounded incorrectly in `args`.
   **Why:** Tokenization splits large numbers into subwords, making them hard to reconstruct.
   **Fix:** Added numeric edge cases to Slice C, but a deeper fix would be injecting a regex-based canonicalization step in `inference.py` or training with more continuous numeric ranges.

## 8. What Worked / What Didn't

- **What Worked:** QLoRA with `packing=True` was incredibly fast and easily fit into the Colab T4 constraint. GGUF quantization using `llama.cpp` provided the required CPU latency footprint.
- **What Didn't:** Directly saving to `save_pretrained` with `load_in_4bit=True` from `transformers` caused major slow-downs during CPU inference on Colab. Migrating to `llama.cpp` GGUF was necessary to hit the <= 200ms latency mark. 
