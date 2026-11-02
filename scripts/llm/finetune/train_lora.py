#!/usr/bin/env python
"""LoRA fine-tune of Qwen3-0.6B on the composer dataset, resumable, for use under the shared GPU lock.

Render rule (device/llm/README.md, results/llm/chat_render_check.json): the prompt is apply_chat_template(..., enable_thinking=False,
add_generation_prompt=True), which ends in <think>\\n\\n</think>\\n\\n; labels are the answer plus <|im_end|>. Embeddings stay tied:
LoRA touches attention and MLP projections only. Each run stops itself at --budget-s, checkpoints, and exits; rerun to continue.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def encode(tok, row, max_len=512):
    msgs = row["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    ans = tok(msgs[-1]["content"] + "<|im_end|>", add_special_tokens=False).input_ids
    ids = (p + ans)[:max_len]
    labels = ([-100] * len(p) + ans)[:max_len]
    return ids, labels


def collate(batch, pad, device):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def answer_loss(model, ids, lab, att):
    """Cross entropy on the answer tokens only. The vocabulary is 151,669 wide, so logits for the whole sequence would need over a
    gigabyte on this shared 8 GB card; the prompt positions never count towards the loss anyway."""
    inner = model.get_base_model()
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    target = lab[:, 1:]
    mask = target != -100
    return F.cross_entropy(inner.lm_head(h[mask]).float(), target[mask])


@torch.no_grad()
def val_loss(model, val, pad, device, bs):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val), bs):
        ids, lab, att = collate(val[i:i + bs], pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = answer_loss(model, ids, lab, att)
        k = (lab[:, 1:] != -100).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "data/llm/finetune/base"))
    ap.add_argument("--train", default=str(REPO / "data/llm/finetune/train.jsonl"))
    ap.add_argument("--val", default=str(REPO / "data/llm/finetune/val.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--budget-s", type=int, default=540)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-limit", type=int, default=120)
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    assert tok.pad_token_id == 151643 and tok.convert_tokens_to_ids("<|im_end|>") == 151645

    rng = random.Random(20261102)
    train = [encode(tok, json.loads(line)) for line in open(a.train)]
    val = [encode(tok, json.loads(line)) for line in open(a.val)][:a.val_limit]
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"step": 0, "log": []}
    if state["step"] >= a.steps:
        print("already done", state["step"])
        return

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, attn_implementation="sdpa")
    base.config.use_cache = False
    if state["step"] > 0:
        model = PeftModel.from_pretrained(base, str(out / "adapter"), is_trainable=True)
    else:
        cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
        model = get_peft_model(base, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable", sum(p.numel() for p in params), "device", device,
          "gpu_mb", torch.cuda.memory_allocated() // 2**20 if device == "cuda" else 0, flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if (out / "opt.pt").exists():
        opt.load_state_dict(torch.load(out / "opt.pt", map_location=device))

    def lr_at(s):
        warm = 40
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))

    def save():
        model.save_pretrained(str(out / "adapter"))
        torch.save(opt.state_dict(), out / "opt.pt")
        state_path.write_text(json.dumps(state, indent=1))

    t0 = time.time()
    model.train()
    rng = random.Random(20261102 + state["step"])
    while state["step"] < a.steps and time.time() - t0 < a.budget_s:
        run_loss = 0.0
        for _ in range(a.accum):
            ids, lab, att = collate(rng.sample(train, a.bs), tok.pad_token_id, device)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = answer_loss(model, ids, lab, att) / a.accum
                loss.backward()
                run_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                print("OOM: checkpointing and stopping; rerun with a smaller --bs", flush=True)
                state["oom"] = True
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                save()
                sys.exit(3)
        for g in opt.param_groups:
            g["lr"] = lr_at(state["step"])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        if state["step"] % 25 == 0:
            print(f"step {state['step']} loss {run_loss:.4f} {(time.time() - t0):.0f}s", flush=True)
        if state["step"] % 250 == 0 or state["step"] == a.steps:
            vl = val_loss(model, val, tok.pad_token_id, device, a.bs)
            state["log"].append({"step": state["step"], "train_loss": round(run_loss, 4), "val_loss": round(vl, 4)})
            print("val", state["log"][-1], flush=True)
            save()
    save()
    print("stopped at", state["step"], "of", a.steps, "peak gpu mb", torch.cuda.max_memory_allocated() // 2**20 if device == "cuda" else 0, flush=True)
    del model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
