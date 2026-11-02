# TwinRead explainer LLM on the UNO Q: runtime notes

Research prototype, not a diagnostic device. An LLM writes the text and it can state wrong facts; `device/explain_check.py` checks an answer against the result it explains.

This file covers the runtime and the model choice. Every number comes from `results/llm/` and was measured on the board. Start
with `results/llm/report_tables.md`; the raw runs are in `results/llm/runs/`.

## What to run

| | |
|---|---|
| Model to ship | `twinread-explain-qwen3-v1-Q4_0-embQ8.gguf`, 419,245,920 bytes, sha256 `84a1d904388270f0b688e1a76632df3d045c4865218346905dfa8ec8ca118bd3`: Qwen3-0.6B fine-tuned on rule-written targets. See "Explaining a result" |
| Base model | `Qwen3-0.6B-Q4_0-embQ8.gguf`, 419,246,784 bytes, sha256 `74d6723293d9969b53051fd50fd985e2c0227178a89715265a11d1d59dea6c51`. All speed measurements below used this file |
| Made from | `unsloth/Qwen3-0.6B-GGUF` `Qwen3-0.6B-Q4_0.gguf`, with only the token embedding re-encoded from Q6_K to Q8_0 |
| Fallback | `bartowski/Qwen2.5-0.5B-Instruct-GGUF` `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397,808,192 bytes, unchanged |
| Runtime | llama.cpp `v0.8.0`, commit `8642541b255bfb5dc39cd351a178be09013a55d6`, static aarch64 build, 14,891,680 bytes |
| Threads | 3 |

On the board the binary is at `device/llm/bin/llama-server` and the models are in `data/llm/`.

```sh
cd ~/twinread-conch-musk-edge
OMP_WAIT_POLICY=passive nice -n 10 device/llm/bin/llama-server \
  -m data/llm/twinread-explain-qwen3-v1-Q4_0-embQ8.gguf -c 1024 -t 3 -tb 3 -np 1 \
  --host 127.0.0.1 --port 1790 --no-webui --cache-ram 0 --ctx-checkpoints 0 --reasoning off
```

Port 1790 is what the benchmarks used; pick whichever suits the app. It binds to loopback and needs no network.

Send each explanation to `POST /v1/chat/completions`, with the body from `device/explain_prompt.py`:

```python
import explain_prompt, explain_check
body = explain_prompt.chat_request(state, audience="doctor")   # state: the /state JSON; audience: "doctor" or "user"
explain_check.check_snapshot(text, state)                      # optional, after the answer: {"passed", "checks", "invented_claim", "details"}
```

`chat_request` raises `explain_prompt.NoDecision` when the state holds no decision. After the server starts, send
`explain_prompt.prewarm_request()` once. It caches the system prompt, so the first click is as fast as every later one (about 21 s
was measured at 200 tokens, T2's prompt). The speed measurements below used T2's earlier prompt, whose text is in
`results/llm/example_results.json`, and `repeat_penalty` 1.1; that prompt's wording and that setting are not what runs now (see
"Explaining a result").

`--reasoning off` turns Qwen3 thinking off for every request. The rendered prompt ends in `<think>\n\n</think>\n\n`, and no run
produced a `<think>` tag or reasoning text.

## How long a user waits (vision model reading at 2 Hz, 3 threads)

| Step | Measured |
|---|---|
| Server load, model already in page cache | 3.3 s |
| Server load, model evicted from page cache first | 4.5 s |
| Prewarm of the system prompt, once per server start | 21.0 s |
| Time to first word, prewarmed | 9.1 to 9.9 s |
| Whole answer, prewarmed (67, 68 and 77 words) | 34.6, 39.3 and 46.0 s |
| Estimated 100-word answer at 1.5 tokens per word | 52.3 to 53.5 s |
| Time to first word with nothing cached (vision idle) | 25.4 s |

The 60 s target is met for 100 words once the prompt is prewarmed. It is not met for a 100-word first click on an unwarmed server,
which adds about 22 s to the first word (31.5 against 9.8 s, measured on the GCC 15 build). Answers near 70 words took about 35 to 40 s.

## Why these settings

**Quantization.** No file is repacked here: llama.cpp's ARM repack needs dotprod or i8mm, and the verbose load log shows
`CPU_Mapped` only, with `REPACK = 1` compiled in but unused. On plain NEON, Q4_0 is the fastest layer format measured. For Qwen3 at
4 threads with no vision: Q4_0 with a Q8_0 embedding 5.36 t/s, with a Q4_0 embedding 5.17, Q4_K_M with a Q8_0 embedding 5.06, Q8_0
everywhere 5.19, Q5_0 with a Q8_0 embedding 4.76, the stock Q4_0 file 4.72, the stock Q4_K_M 4.40. The stock files keep the shared
input/output embedding in Q6_K, and that tensor is read in full for every generated token, so changing only it gave most of the gain.

**Compiler.** GCC 14 beat GCC 15.2 on this workload: 4.15 to 4.21 t/s against 4.00 to 4.05, with prompt speed 11.3 to 11.4 against
10.3 to 10.4 t/s. GCC 14 with `-mtune=cortex-a53` gave the best run (4.21). LTO, tuning and OpenMP versus ggml's own thread pool made
no measurable difference on GCC 15. The official b11960 arm64 build ran without any crash at 4.15 to 4.19 t/s, so it is a valid
option, but it needs its own shared libraries plus the board's `libgomp` and `libstdc++`.

**Threads, priority, wait policy.** With the vision model reading at 2 Hz:

| LLM setting | gen t/s | read p50 / p95 / max ms | peak C |
|---|---|---|---|
| 4 threads, nice 0 | 3.11 | 105.5 / 146.3 / 218.2 | 67.9 |
| 4 threads, nice 10 | 3.82 | 53.6 / 103.7 / 136.9 | 69.2 |
| 3 threads, nice 0 | 2.90 | 105.3 / 157.1 / 227.0 | 65.4 |
| 3 threads, nice 10 | 2.94 | 53.8 / 107.7 / 147.3 | 65.1 |
| 3 threads, nice 10, passive wait | 3.36 | 53.6 / 105.4 / 142.3 | 64.8 |
| 3 threads, nice 10, passive wait, GCC 14 build (final) | 3.48 | 51.8 / 104.2 / 136.5 | 63.5 |
| 4 threads, nice 10, passive wait, GCC 14 build | 4.28 | 49.7 / 71.5 / 121.0 | 68.2 |
| 2 threads, nice 0 | 2.72 | 60.4 / 123.3 / 152.5 | 62.6 |

The rows before the "GCC 14 build" ones used the GCC 15 binary. Reads alone take 48.9 ms at p50, and no read arrived more than 100 ms
late in any tandem run. Use nice 10: it halved read latency and did not slow the LLM (it sped it up at 4 threads). Nice 19 gave no clear
gain over nice 10. Passive wait gained 10 to 14 % in tandem and cost 4 % with no vision running (5.15 against 5.36 t/s at 4 threads,
3.87 against 4.04 at 3 threads).

**Memory.** RSS is 612 MB, of which 207 MB is private and the rest is the memory-mapped model, which the kernel can drop and re-read.
The vision process adds 85 MB and the app in standby 58 MB. In tandem the lowest available memory was 2,831 to 2,840 MB of 3,667 MB,
and swap was never used in any run. The kiosk browser ran in only 2 of the 75 runs; it cost about 220 MB, an estimate from baselines
rather than a tandem measurement. `-c 640 -b 128 -ub 128` saves 47 MB with the same generation speed and a slightly slower prompt (10.0 against 10.4 t/s). Disabling mmap (`--load-mode none`) gave
no speed gain and made 600 MB private, so keep mmap.

**Idle.** A loaded server used 0.0 to 0.05 % of a core. Keeping it loaded costs memory, not heat.

**Heat, hottest zone.** A rested board reads about 43 to 46 C. The kernel's first passive trip is 90 C and the clock stayed at 2,016 MHz in every run.

| Load | peak C |
|---|---|
| Vision at 2 Hz only, 126 s | 52.2 |
| One prewarmed explanation in the final setup, vision at 2 Hz | 62.9 |
| Back-to-back explanations, final setup, 156 s | 64.5 |
| Back-to-back explanations, 4 threads, passive, vision at 2 Hz, 176 s | 69.2 |
| Back-to-back explanations, 4 threads, GCC 15, active wait, no vision | 72.3, killed by the 72 C guard at 123 s |
| Back-to-back explanations, 2 threads, no vision, 175 s | 58.5 |

The board cools quickly: a median window ended at 62.6 C and read 47.8 C at the next check, 30 to 60 s later or after a model push.
No window here lasted beyond about 3 minutes, so a longer sustained run is untested.

## Rebuilding

```sh
sudo apt install g++-14-aarch64-linux-gnu cmake ninja-build
scripts/llm/build_llama_aarch64.sh --deploy    # default variant gcc14a53; also builds x86 llama-quantize at the same commit
```

Scripts: `scripts/llm/bench_on_board.py` (the benchmark, stdlib only), `scripts/llm/board.py` (lock, cool-down, model push, plans in
`scripts/llm/plans/`), `scripts/llm/vision_loop.py`, `scripts/llm/report_tables.py`. Run hot work through `board.py`: it holds
`~/.config/twinread/board.lock` and refuses to start above the temperature limit.

## Fine-tuning and converting the result

- Base model: `Qwen/Qwen3-0.6B` (Apache-2.0, ungated, `Qwen3ForCausalLM`, 28 layers, hidden 1024, tied embeddings). Fallback base:
  `Qwen/Qwen2.5-0.5B-Instruct`.
- Tokenizer: `Qwen2Tokenizer`. EOS `<|im_end|>` 151645, pad `<|endoftext|>` 151643, `<think>` 151667, `</think>` 151668.
- Render with `enable_thinking=False`. Checked in `results/llm/chat_render_check.json`: the training text is exactly the inference prompt
  (ending in `<think>\n\n</think>\n\n`), then the answer, then `<|im_end|>\n`. So the labels are the answer plus `<|im_end|>`.
- Keep embeddings tied. A merged model saved with `save_pretrained(..., safe_serialization=True)` had no `lm_head` in safetensors and no
  separate `output.weight` in the GGUF. A separate output tensor would add its size to the file and slow generation.
- Convert and quantize at the pinned commit (`build/llama.cpp-v0.8.0`, from the build script):

```sh
python build/llama.cpp-v0.8.0/convert_hf_to_gguf.py MERGED_DIR --outfile merged-f16.gguf --outtype f16
build/llama.cpp-v0.8.0/build-host/bin/llama-quantize --token-embedding-type q8_0 merged-f16.gguf merged-Q4_0-embQ8.gguf Q4_0
```

  The flag is required: the default for Qwen3 is a Q6_K embedding, which was slower. Check the result with `scripts/llm/inspect_gguf.py`:
  `token_embd.weight` should be Q8_0 and `separate_output_tensor` false.
- That path was first proven on tiny random-weight models with the real tokenizers (`results/llm/conversion_check*.json`), then on real weights
  (`results/llm/finetune/conversion_check_real.json`). A base conversion had the same 310 tensor names, shapes and types as the base file and no
  output tensor. Layer bytes differ because the base file was requantized from a Q4_0 download; quantizing from f16 or bf16 gave identical bytes.
  Next-token agreement with PyTorch along each file's own greedy text: 0.905 for the conversion, 0.909 for the base file, 0.962 for the fine-tune.
  The fine-tuned GGUF renders the same token IDs as training on the pinned server (`results/llm/finetune/render_check_finetuned.json`).
- Keep the system prompt's bytes stable. The cache only reuses an exact prefix (201 to 210 of about 290 tokens here).

## Explaining a result

`device/explain_prompt.py` builds the prompt and `device/explain_check.py` checks the answer. Both are stdlib only. The heads are always Head A and Head B.
The generated text carries no disclaimers; "the model read this tile as..." and the one-tile limit carry that meaning.

```python
body = explain_prompt.chat_request(state, audience="doctor", seed=None, stream=True)   # or audience="user"
result = explain_check.check_snapshot(text, state)                                     # result["passed"], ["checks"], ["invented_claim"], ["details"]
fallback = explain_prompt.facts_text(state)                                            # correct by construction, no LLM
```

Settings are in `explain_prompt.SAMPLING`: temperature 0.3, top_p 0.9, repeat_penalty 1.0, max_tokens 170. Do not raise repeat_penalty: it also
penalises digits and words in the facts, and 1.1 lowered validation pass rates on every prompt tried (`results/llm/finetune/prompt_dev_log.json`).

Evaluation is in `results/llm/finetune/`, pre-registered in `preregistration.md`, with totals in `summary.json`. On 200 scenario-balanced held-out prompts on
x86, at the registered seed:

| Arm | Passed every check | Invented claims | Natural-mix pass |
|---|---|---|---|
| Base + context | 177 (0.885) | 5 | 0.891 |
| Fine-tuned + same context | 195 (0.975) | 4 | 0.976 |
| Rule-based composer | 200 (1.000, by construction) | 0 | 1.000 |

McNemar base against fine-tuned: 5 against 23, p = 0.0009, so the fine-tune ships by the rule. What that does and does not show:

- **The gain is on repeated prompts.** It comes from agree and no_tissue. On review and disagree there was no difference, nor on the 95 prompts never seen in
  training (91 against 92). Agree is 88 % of natural tiles, so the gain is real in use, but the fine-tune does not generalise better on new facts.
- **Different errors over 600 answers.** Base never misstated a number, but it dropped the review-head reason in 108 of 141 answers with two reasons, and said
  what the image shows in 17. The fine-tune did neither, but stated a wrong Head B confidence in 3, for example 70 % where the fact is 100 %. Those 3 fail `explain_check`.
- **The template is as good and faster.** A blind judge on 60 items gave readability 4.25 (fine-tune), 4.10 (composer), 3.65 (base), fine-tune against composer p = 0.35,
  and accuracy 49, 53 and 35 of 60. An LLM answer takes 31 to 37 s on the board; the composer took 0.66 ms on the PC (not timed on the board).

On the board, with the fine-tune and no vision load (`results/llm/finetune/board_check.json`): load 3.4 s, prewarm 18.5 s, first word 9.2 to 10.7 s, 4.00 to 4.07 t/s,
whole answer 30.8 to 37.3 s for 64 to 72 words, peak 60.4 C. For runs beside the vision model at 2 Hz, including the camera and a browser, see
`results/uno_q/onboard/explainer/` and `device/README.md`.

`device/explain_check.py` now checks each answer more strictly than the registered copy: the combined percentage by position, every flag reason, and a
word list built by `scripts/llm/finetune/build_vocab.py`. On the stored held-out answers it wrongly rejected none of the fine-tune's and passed 588 of 600
(`results/llm/finetune/runtime_checker_eval.json`). This is exploratory and leaves the pre-registered result unchanged. The word list comes from the fine-tune's
training text, so this version cannot compare the base model and the fine-tune.

The fine-tuned weights derive from numeric outputs of `bundles/cm` (CC-BY-NC-ND), so they stay local until the owner decides about any release.

## Known limits

- With T2's first prompt, Qwen3 gave sensible English and mostly right numbers but still slipped, for example saying the
  fused value "indicates disagreement" when the heads agree. See "Explaining a result" for the current prompt.
- The fallback's answers are faster (5.3 t/s in tandem, 491 MB RSS) and noticeably less accurate: it called MUS "mucus" and swapped heads.
- Qwen2.5's "Q4_K_M" file is mostly Q5_0 and Q8_0, because hidden size 896 does not fit K-quant blocks.
- The vision loop used a stored 224 px image, so there was no camera or resize cost. Real camera frames will cost more.
- `device/llm/bin/` is not git-ignored in this repo.
