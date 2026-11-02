# Pre-registration: does fine-tuning beat plain context for the TwinRead explainer?

Written 2026-11-02, before any output from any fine-tuned checkpoint and before any output from any arm on the held-out or edge tiles.
Hashes of the frozen files are in `preregistered_hashes.json`.

## Question

Qwen3-0.6B explains one TwinRead result in text. Does a LoRA fine-tune on composer targets beat the base model given the same context?
A negative result is an acceptable outcome.

## Held-out set

- `split.json`, seed 20261102, split by tile before anything was composed.
- 200 prompts: 50 tiles per situation (agree, review, disagree, no_tissue), audience alternating doctor and user within each situation.
- Natural situation rates are in `natural_rates.json` (agree 0.879, review 0.103, disagree 0.017, no_tissue 0.001). The natural-mix figure
  is the per-situation pass rate weighted by those rates. It is a reweighting, not a separate sample.
- Limit: blank and dark frames are constructed on this PC and passed through `crop_from_frame` and `infer`, since the camera stays closed.
  One no_tissue tile in the held-out set or more comes from natural CRC-VAL-HE-7K data.

## Arms (identical prompts from `device/explain_prompt.py`, identical settings)

1. **base**: `data/llm/gen/Qwen3-0.6B-Q4_0-embQ8.gguf`, sha256 `74d67232...`, the file T2 measured on the board.
2. **finetuned**: the final quantized GGUF, not a PyTorch checkpoint. Its sha256 is added below before its first held-out run.
3. **composer**: `scripts/llm/finetune/composer.py` itself as a control. It passes by construction, so it is reported, not tested against
   the primary metric.
4. Sensitivity only, no bearing on the rule: **realbase**, the same base weights put through the conversion path used for the fine-tune.
   T2's file was requantized from a Q4_0 download, so about a quarter of its layer bytes differ from a direct conversion.

Settings: `temperature 0.3, top_p 0.9, repeat_penalty 1.0, max_tokens 170`, seed = base seed + item position, pinned llama.cpp commit,
host x86 server, 3 or 4 threads. `repeat_penalty 1.0` replaced T2's 1.1 after validation runs on the base model only showed 1.1 corrupting
copied digits; the same setting applies to every arm.

## Primary metric and test

- **Primary metric:** share of held-out answers that pass every check in `device/explain_check.py`, at base seed 1, n = 200 per arm.
- **Test:** exact two-sided McNemar test on base against finetuned, paired by item, alpha 0.05.

## Ship rule

The fine-tuned model ships only if **all** hold at base seed 1:

1. its pass share is higher than base's;
2. the McNemar p-value is below 0.05;
3. its invented-claim count is no higher than base's (`invented_claim` in `explain_check`: stage, grade, prognosis, treatment, patient,
   whole-slide, diagnosis or image claims, a class not in the result, or a number not in the result).

Otherwise base + context ships, and the report says so plainly. The rule compares the two LLMs only. Whether either should ship instead of
the composer is a separate question for the owner, answered with the control and the judge data.

## Secondary analyses, none of which can change the rule

- Base seeds 1001 and 2001 for base and finetuned, plus a per-item sign-flip test pooled over all three seeds.
- Per situation, per audience, per check, natural mix, Wilson 95 % intervals.
- Prompts whose exact user turn never appeared in training (63 of 252 possible held-out user turns do appear in training, because rounded
  facts repeat, for instance TUM at 100 % three times).
- An edge set of 11 tiles where the combined call differs from what the heads named. Reported separately.
- Base against realbase on the same items.
- Blind judge (`claude -p --model sonnet`), 15 held-out items per situation = 60 items, three texts each (base, finetuned, composer) in a per-item
  random order labelled A/B/C: accuracy verdict, readability 1-5, usefulness 1-5, best pick. Paired sign-flip tests on the ratings. The judge output
  never enters a training set.
- Hand read of at least 12 answers per arm, with a label of whether the text is factually correct.

## Known threats, stated in advance

- **The checker and the targets share vocabulary.** I wrote both, and the composer's phrasings are all known to the checker. A fine-tune that
  repeats composer sentences is favoured over a base model that is right in phrasing the checker does not know. The judge's accuracy verdict and my
  hand read exist to measure this: the report gives the checker's disagreement with each, per arm.
- **The checker was adjusted while reading validation outputs**, mostly base outputs, which moves it toward base's phrasings, plus a small smoke fine-tune
  (40 validation prompts). Nothing from the held-out set was seen. It is now frozen. A checker bug found later would be reported beside the registered
  score, and the rule stays on the registered score.
- **The prompt was tuned on the base model only**, on the validation split, from pass 0.00 to 0.92. That favours base, which is intended: the comparison
  should be against the best plain context.
- **Host and board differ.** Evaluation runs on x86; only a short check runs on the board, so outputs there can differ in small ways.
- **no_tissue prompts repeat.** Tissue percentage is the only varying fact, so many are identical. Per-item seeds make them replicates of sampling
  rather than of tiles, and the number of distinct prompts per situation is reported.

## Fine-tune selection

Hyperparameters and the final checkpoint are chosen on the validation split only. The number of candidates tried is reported. Training uses train tiles only.

## Addendum: finetuned GGUF

Added 2026-11-02T09:41:20, before any held-out output from this arm, and with no held-out output from any arm yet read or scored:

- File: `data/llm/finetune/twinread-explain-qwen3-v1-Q4_0-embQ8.gguf`, 419245920 bytes, sha256 `84a1d904388270f0b688e1a76632df3d045c4865218346905dfa8ec8ca118bd3`.
- LoRA r=16 alpha=32 on attention and MLP projections, 1000 optimizer steps, batch 2 with accumulation 2, learning rate 3e-4 cosine, 8,000 composer rows on train tiles.
- Candidates tried: this one, plus one 150-step smoke run on an older prompt that was deleted. Selection used the validation split only:
  this file passed 0.983 of 175 validation prompts (invented 0.017), base 0.920 (invented 0.029), in `prompt_dev_log.json` once summarised.
- Base held-out runs were generated earlier and were not read: only their line counts were looked at.
