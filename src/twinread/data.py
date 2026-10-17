"""Data access for TwinRead: NCT-CRC-HE (9 tissue classes) and Camelyon17-WILDS (5 hospitals).

Everything is stored as plain ``.npy`` arrays plus a JSON manifest so later tasks (and NCSA Delta jobs) never touch
parquet again.  Remote parquet shards are read with HTTP range requests, one 100-row group at a time, so only the
row groups that are needed are transferred and no shard is stored on disk.  Downloaded bytes are untrusted data and
are only decoded as images (PIL); nothing from them is executed.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CRC_CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
CRC_REPO = "1aurent/NCT-CRC-HE"
CAM_REPO = "wltjr1007/Camelyon17-WILDS"
CAM_TEST_HOSPITAL = 2  # WILDS test centre
CAM_HOSPITALS = 5
_RETRIES = 4


# ----------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_local(shard) -> bool:
    return isinstance(shard, Path) or Path(str(shard)).exists()


def _open(shard):
    """Open a local path or an ``datasets/<repo>/data/<file>`` Hub path as a binary file object."""
    if _is_local(shard):
        return open(shard, "rb")
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(str(shard))


def _name(shard) -> str:
    return Path(str(shard)).name


def _list_remote(repo: str, prefixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    from huggingface_hub import HfApi
    out = []
    for e in HfApi().list_repo_tree(repo, path_in_repo="data", repo_type="dataset"):
        n = Path(e.path).name
        if n.endswith(".parquet") and n.startswith(prefixes) and not n.startswith(exclude):
            out.append(f"datasets/{repo}/{e.path}")
    return sorted(out)


def _decode(b: bytes, size: int) -> np.ndarray:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def split_val(y: np.ndarray, frac: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (train_idx, val_idx).  Every class with >= 2 members puts >= 1 in val and keeps >= 1 in train."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        nv = 0 if len(idx) < 2 else min(len(idx) - 1, max(1, int(round(frac * len(idx)))))
        va.append(idx[:nv])
        tr.append(idx[nv:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


# ----------------------------------------------------------------------------------------------------------------
# parquet scanning / sampling
# ----------------------------------------------------------------------------------------------------------------
def _scan(shards, key_cols: tuple[str, ...]):
    """Read only the small key columns of every shard.

    Returns (groups, n_rows) where ``groups[key][(shard_idx, row_group)] = [row-in-group, ...]``.
    ``key`` is the value of key_cols (scalar for one column, tuple for several).
    """
    groups: dict = {}
    total = 0
    for si, shard in enumerate(shards):
        with _open(shard) as f:
            pf = pq.ParquetFile(f)
            tbl = pf.read(columns=list(key_cols))
            cols = [tbl[c].to_numpy() for c in key_cols]
            start = 0
            for rg in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                keys = list(zip(*[c[start:start + n].tolist() for c in cols]))
                for r, k in enumerate(keys):
                    k = k[0] if len(key_cols) == 1 else k
                    groups.setdefault(k, {}).setdefault((si, rg), []).append(r)
                start += n
            total += start
    return groups, total


def _plan_groups(groups: dict, quotas: dict, rng: np.random.Generator, rows_per_group: int | None = None) -> dict:
    """Choose row groups at random until each key's quota is met, then trim to exactly the quota.

    ``groups`` is the ``_scan`` structure.  Returns ``{(shard_idx, row_group): sorted rows}``.
    Sampling whole row groups (instead of single rows) keeps the number of transferred bytes close to the number of
    bytes kept; ``rows_per_group`` caps how many rows one group may contribute (more groups -> more slide diversity).
    """
    plan: dict = {}
    for key in sorted(quotas, key=str):
        quota = quotas[key]
        gmap = groups.get(key, {})
        order = sorted(gmap)
        order = [order[i] for i in rng.permutation(len(order))]
        picked, have = [], 0
        for g in order:
            if have >= quota:
                break
            rows = gmap[g]
            if rows_per_group is not None and len(rows) > rows_per_group:
                rows = sorted(rng.choice(rows, rows_per_group, replace=False).tolist())
            picked.extend((g, r) for r in rows)
            have += len(rows)
        if len(picked) > quota:
            keep = sorted(rng.choice(len(picked), quota, replace=False).tolist())
            picked = [picked[i] for i in keep]
        for g, r in picked:
            plan.setdefault(g, set()).add(r)
    return {g: sorted(rs) for g, rs in sorted(plan.items())}


def _read_plan(shards, plan: dict, extra_cols: tuple[str, ...], size: int, out_x: np.ndarray):
    """Decode the planned rows into ``out_x`` (in plan order).  Returns {col: array} for ``extra_cols``."""
    extras = {c: [] for c in extra_cols}
    pos = 0
    by_shard: dict = {}
    for (si, rg), rows in plan.items():
        by_shard.setdefault(si, []).append((rg, rows))
    for si in sorted(by_shard):
        f = _open(shards[si])
        pf = pq.ParquetFile(f)
        for rg, rows in by_shard[si]:
            for attempt in range(_RETRIES):
                try:
                    tbl = pf.read_row_group(rg, columns=["image", *extra_cols]).take(rows)
                    break
                except Exception:  # transient network error: reopen and retry
                    if attempt == _RETRIES - 1 or _is_local(shards[si]):
                        raise
                    time.sleep(2 * (attempt + 1))
                    f.close()
                    f = _open(shards[si])
                    pf = pq.ParquetFile(f)
            for i, im in enumerate(tbl["image"].to_pylist()):
                out_x[pos + i] = _decode(im["bytes"], size)
            for c in extra_cols:
                extras[c].append(tbl[c].to_numpy())
            pos += len(rows)
        f.close()
    assert pos == len(out_x)
    return {c: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for c, v in extras.items()}


def _write_shuffled(x_mm: np.ndarray, y: np.ndarray, rng, out_x: Path, out_y: Path, extra: dict | None = None):
    """Shuffle rows (seeded) while copying from a scratch memmap to the final npy; returns nothing."""
    perm = rng.permutation(len(y))
    final = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.uint8, shape=x_mm.shape)
    for s in range(0, len(perm), 512):
        final[s:s + 512] = x_mm[perm[s:s + 512]]
    final.flush()
    del final
    np.save(out_y, y[perm].astype(np.int64))
    return perm


def _fetch_images(shards, plan, extra_cols, size, out_x: Path, out_y: Path, label_col: str, seed: int):
    """plan -> shuffled ``out_x``/``out_y`` npy files.  Returns (y, extras_in_final_order)."""
    n = sum(len(r) for r in plan.values())
    scratch = out_x.with_suffix(".scratch.npy")
    mm = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    ex = _read_plan(shards, plan, (label_col, *extra_cols), size, mm)
    mm.flush()
    y = ex[label_col].astype(np.int64)
    perm = _write_shuffled(mm, y, np.random.default_rng(seed + 1), out_x, out_y)
    del mm
    scratch.unlink()
    return y[perm], {c: ex[c][perm] for c in extra_cols}


# ----------------------------------------------------------------------------------------------------------------
# NCT-CRC-HE
# ----------------------------------------------------------------------------------------------------------------
def _counts(y: np.ndarray) -> dict:
    return {c: int((y == i).sum()) for i, c in enumerate(CRC_CLASSES)}


def fetch_crc(out_dir: Path, n_train: int | None = 20000, seed: int = 0, *, _shards=None, _test_shards=None) -> dict:
    """Stratified ``n_train`` tiles from NCT-CRC-HE-100K (``None`` = all 100k) + the full CRC-VAL-HE-7K as test.

    Writes ``crc_{train,test}_{x,y}.npy`` and ``crc_manifest.json`` into ``out_dir``.  Train sampling is by random
    100-row parquet groups per class (shards are class-sorted), trimmed to exactly ceil(n_train/9) per class.
    ``_shards`` / ``_test_shards`` inject local parquet files (tests).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_shards = list(_shards) if _shards is not None else _list_remote(CRC_REPO, ("NCT_CRC_HE_100K-",))
    if _test_shards is not None:
        test_shards = list(_test_shards)
    elif _shards is not None:
        test_shards = list(_shards)
    else:
        test_shards = _list_remote(CRC_REPO, ("CRC_VAL_HE_7K-",))
    rng = np.random.default_rng(seed)

    groups, _ = _scan(train_shards, ("label",))
    if n_train is None:
        quotas = {c: sum(len(r) for r in groups.get(c, {}).values()) for c in range(9)}
    else:
        q = math.ceil(n_train / 9)
        quotas = {c: min(q, sum(len(r) for r in groups.get(c, {}).values())) for c in range(9)}
    plan = _plan_groups(groups, quotas, rng)
    ytr, _ = _fetch_images(train_shards, plan, (), 224, out_dir / "crc_train_x.npy", out_dir / "crc_train_y.npy",
                           "label", seed)

    tgroups, _ = _scan(test_shards, ("label",))
    tquotas = {c: sum(len(r) for r in tgroups.get(c, {}).values()) for c in range(9)}
    tplan = _plan_groups(tgroups, tquotas, rng)
    yte, _ = _fetch_images(test_shards, tplan, (), 224, out_dir / "crc_test_x.npy", out_dir / "crc_test_y.npy",
                           "label", seed)

    manifest = {
        "source": f"hf:{CRC_REPO}" if _shards is None else "local-shards",
        "shards": [_name(s) for s in train_shards],
        "test_shards": [_name(s) for s in test_shards],
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "class_counts": _counts(ytr), "class_counts_test": _counts(yte),
        "classes": list(CRC_CLASSES), "seed": seed,
        "sha256": {f: _sha256(out_dir / f) for f in
                   ("crc_train_x.npy", "crc_train_y.npy", "crc_test_x.npy", "crc_test_y.npy")},
    }
    (out_dir / "crc_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def crc_from_folder(root: Path, out_dir: Path, split: str) -> dict:
    """Build ``crc_<split>_{x,y}.npy`` from NCT-CRC class folders (``<root>/<CLASS>/*.tif``, NCSA Delta layout).

    ``split`` is ``"train"`` or ``"test"``.  The manifest ``crc_manifest.json`` is created or updated in place
    (``n_<split>``, ``class_counts`` for train / ``class_counts_test`` for test, ``sha256``).
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be train or test, got {split!r}")
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    bad = [d.name for d in dirs if d.name not in CRC_CLASSES]
    if bad:
        raise ValueError(f"unexpected class folders {bad}; expected {CRC_CLASSES}")
    files, labels = [], []
    for d in dirs:
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg"))
        files += fs
        labels += [CRC_CLASSES.index(d.name)] * len(fs)
    if not files:
        raise ValueError(f"no images under {root}")
    xp, yp = out_dir / f"crc_{split}_x.npy", out_dir / f"crc_{split}_y.npy"
    x = np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8, shape=(len(files), 224, 224, 3))
    for i, p in enumerate(files):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.BICUBIC)
            x[i] = np.asarray(im, dtype=np.uint8)
    x.flush()
    del x
    y = np.asarray(labels, dtype=np.int64)
    np.save(yp, y)
    mp = out_dir / "crc_manifest.json"
    m = json.loads(mp.read_text()) if mp.exists() else {"classes": list(CRC_CLASSES)}
    m[f"n_{split}"] = int(len(y))
    m["class_counts" if split == "train" else "class_counts_test"] = _counts(y)
    m[f"source_{split}"] = f"folder:{root}"
    m.setdefault("sha256", {}).update({xp.name: _sha256(xp), yp.name: _sha256(yp)})
    mp.write_text(json.dumps(m, indent=2))
    return m


# ----------------------------------------------------------------------------------------------------------------
# Camelyon17-WILDS
# ----------------------------------------------------------------------------------------------------------------
def fetch_camelyon(out_dir: Path, per_hospital: int = 1500, seed: int = 0, *, _shards=None) -> dict:
    """``per_hospital`` 96x96 tiles from each of the 5 WILDS hospitals, half normal / half tumor where available.

    Writes ``cam_{x,y,h}.npy`` and ``cam_manifest.json``.  ``h`` is the WILDS ``center`` id (0-4; 2 = WILDS test).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = list(_shards) if _shards is not None else _list_remote(CAM_REPO, ("train-", "validation-", "test-"))
    groups, _ = _scan(shards, ("center", "label"))
    half = per_hospital // 2
    quotas = {}
    for h in range(CAM_HOSPITALS):
        avail1 = sum(len(r) for r in groups.get((h, 1), {}).values())
        avail0 = sum(len(r) for r in groups.get((h, 0), {}).values())
        n1 = min(per_hospital - half, avail1)
        n0 = min(half, avail0)
        # if one label is short, give the slack to the other
        n0 = min(avail0, per_hospital - n1) if n1 < per_hospital - half else n0
        quotas[(h, 0)], quotas[(h, 1)] = n0, n1
    plan = _plan_groups(groups, quotas, np.random.default_rng(seed), rows_per_group=25)
    y, ex = _fetch_images(shards, plan, ("center",), 96, out_dir / "cam_x.npy", out_dir / "cam_y.npy", "label", seed)
    h = ex["center"].astype(np.int64)
    np.save(out_dir / "cam_h.npy", h)
    manifest = {
        "source": f"hf:{CAM_REPO}" if _shards is None else "local-shards",
        "shards": [_name(s) for s in shards],
        "per_hospital": per_hospital, "n": int(len(y)), "seed": seed,
        "counts": {str(k): {"normal": int(((h == k) & (y == 0)).sum()), "tumor": int(((h == k) & (y == 1)).sum())}
                   for k in range(CAM_HOSPITALS)},
        "sampling": "random 100-row parquet groups per (hospital,label), <=25 rows per group, trimmed to quota",
        "sha256": {f: _sha256(out_dir / f) for f in ("cam_x.npy", "cam_y.npy", "cam_h.npy")},
    }
    (out_dir / "cam_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ----------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------
def load_split(out_dir: Path, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) for ``dataset`` in {crc, cam} and ``split`` in {train, val, test}.

    crc: test = CRC-VAL-HE-7K (never mixed with train); val = stratified 10% of the saved train subset.
    cam: test = hospital 2 (WILDS test centre); train/val = stratified (label x hospital) 90/10 of hospitals 0,1,3,4.
    """
    out_dir = Path(out_dir)
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    if dataset == "crc":
        if split == "test":
            return np.load(out_dir / "crc_test_x.npy"), np.load(out_dir / "crc_test_y.npy")
        x = np.load(out_dir / "crc_train_x.npy", mmap_mode="r")
        y = np.load(out_dir / "crc_train_y.npy")
        tr, va = split_val(y, 0.1, 0)
        idx = tr if split == "train" else va
        return np.ascontiguousarray(x[idx]), y[idx]
    if dataset == "cam":
        x = np.load(out_dir / "cam_x.npy", mmap_mode="r")
        y = np.load(out_dir / "cam_y.npy")
        h = np.load(out_dir / "cam_h.npy")
        if split == "test":
            idx = np.flatnonzero(h == CAM_TEST_HOSPITAL)
        else:
            rest = np.flatnonzero(h != CAM_TEST_HOSPITAL)
            tr, va = split_val(y[rest] * CAM_HOSPITALS + h[rest], 0.1, 0)
            idx = rest[tr if split == "train" else va]
        return np.ascontiguousarray(x[idx]), y[idx]
    raise ValueError(f"unknown dataset {dataset!r}")
