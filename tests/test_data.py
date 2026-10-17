import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from twinread import data
from twinread.data import CRC_CLASSES, crc_from_folder, fetch_crc, load_split, split_val

FIX = Path(__file__).parent / "fixtures" / "mini_crc.parquet"


def _fetch(tmp_path, **kw):
    test_shard = tmp_path / "CRC_VAL_HE_7K-00000-of-00001-mini.parquet"
    shutil.copy(FIX, test_shard)
    return fetch_crc(tmp_path, n_train=kw.pop("n_train", 20000), _shards=[FIX], _test_shards=[test_shard], **kw)


def test_split_val_stratified():
    y = np.repeat(np.arange(9), 30)
    tr, va = split_val(y, 0.1, 0)
    assert set(y[va]) == set(range(9))
    assert not set(tr) & set(va)
    assert len(tr) + len(va) == len(y)
    assert (np.bincount(y[va]) == 3).all()


def test_split_val_tiny_class_still_in_val():
    y = np.array([0] * 20 + [1] * 2)
    tr, va = split_val(y, 0.1, 0)
    assert 1 in set(y[va]) and 1 in set(y[tr])


def test_fetch_crc_from_local_shards(tmp_path):
    m = _fetch(tmp_path)
    x = np.load(tmp_path / "crc_train_x.npy")
    y = np.load(tmp_path / "crc_train_y.npy")
    assert x.dtype == np.uint8 and x.shape == (27, 224, 224, 3)
    assert y.dtype == np.int64 and y.shape == (27,)
    assert sum(m["class_counts"].values()) == len(y) == m["n_train"]
    assert set(m["class_counts"]) == set(CRC_CLASSES)
    # decoded pixels match their label (red channel was class-coded)
    assert (x[..., 0][:, 0, 0] == y * 25).all()
    assert json.loads((tmp_path / "crc_manifest.json").read_text())["sha256"]["crc_train_x.npy"]


def test_fetch_crc_stratified_quota_and_seed(tmp_path):
    m = _fetch(tmp_path, n_train=18)  # 2 per class
    assert m["n_train"] == 18 and set(m["class_counts"].values()) == {2}
    y1 = np.load(tmp_path / "crc_train_y.npy")
    m2 = _fetch(tmp_path, n_train=18)
    assert (np.load(tmp_path / "crc_train_y.npy") == y1).all()
    assert m2["sha256"] == m["sha256"]


def test_fetch_crc_n_train_none_takes_all(tmp_path):
    m = _fetch(tmp_path, n_train=None)
    assert m["n_train"] == 27


def test_load_split_test_is_crc_val(tmp_path):
    m = _fetch(tmp_path)
    assert all("CRC_VAL" in s for s in m["test_shards"])
    assert not any("CRC_VAL" in s for s in m["shards"])
    xt, yt = load_split(tmp_path, "crc", "test")
    assert xt.shape[0] == 27 == len(yt)
    xtr, ytr = load_split(tmp_path, "crc", "train")
    xv, yv = load_split(tmp_path, "crc", "val")
    assert len(ytr) + len(yv) == 27 and set(yv) == set(range(9))


def test_crc_from_folder(tmp_path):
    root = tmp_path / "NCT-CRC-HE-100K" / "data-samples"
    rng = np.random.default_rng(0)
    for ci in (1, 8):
        d = root / CRC_CLASSES[ci]
        d.mkdir(parents=True)
        for k in range(3):
            a = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(a).save(d / f"{CRC_CLASSES[ci]}-{k}.tif")
    out = tmp_path / "out"
    m = crc_from_folder(root, out, "test")
    x, y = np.load(out / "crc_test_x.npy"), np.load(out / "crc_test_y.npy")
    assert x.shape == (6, 224, 224, 3) and x.dtype == np.uint8 and y.dtype == np.int64
    assert sorted(set(y.tolist())) == [1, 8]
    assert m["n_test"] == 6 and m["class_counts_test"]["TUM"] == 3
    assert json.loads((out / "crc_manifest.json").read_text())["n_test"] == 6


def test_crc_from_folder_rejects_unknown_class(tmp_path):
    (tmp_path / "root" / "FOO").mkdir(parents=True)
    with pytest.raises(ValueError):
        crc_from_folder(tmp_path / "root", tmp_path / "o", "train")


def test_cluster_plan_respects_quota_and_seed():
    # groups: key -> {(shard, rg): [rows]}
    groups = {0: {(0, g): list(range(100)) for g in range(10)}}
    a = data._plan_groups(groups, {0: 250}, np.random.default_rng(0))
    b = data._plan_groups(groups, {0: 250}, np.random.default_rng(0))
    assert sum(len(r) for r in a.values()) == 250 and a == b
    assert len(data._plan_groups(groups, {0: 5000}, np.random.default_rng(0))) == 10


def _mini_cam(path, n_per=(40, 40)):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(1)
    imgs, lab, cen = [], [], []
    for h in range(5):
        for lbl in (0, 1):
            for _ in range(n_per[lbl]):
                a = rng.integers(0, 256, (96, 96, 4), dtype=np.uint8)
                a[..., 0] = h * 40 + lbl * 10
                b = io.BytesIO()
                Image.fromarray(a, "RGBA").save(b, format="PNG")
                imgs.append({"bytes": b.getvalue(), "path": "x.png"})
                lab.append(lbl)
                cen.append(h)
    t = pa.table({"image": pa.array(imgs, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                  "label": pa.array(lab, type=pa.int64()), "center": pa.array(cen, type=pa.int64())})
    pq.write_table(t, path, row_group_size=20)


def test_fetch_camelyon_local_and_load_split(tmp_path):
    shard = tmp_path / "train-00000-of-00001.parquet"
    _mini_cam(shard)
    m = data.fetch_camelyon(tmp_path, per_hospital=20, _shards=[shard])
    x, y, h = (np.load(tmp_path / f"cam_{k}.npy") for k in "xyh")
    assert x.shape == (100, 96, 96, 3) and x.dtype == np.uint8 and m["n"] == 100
    assert sorted(set(h.tolist())) == [0, 1, 2, 3, 4]
    assert all(v == {"normal": 10, "tumor": 10} for v in m["counts"].values())
    assert (x[..., 0][:, 0, 0] == h * 40 + y * 10).all()  # pixels stayed aligned with labels after shuffling
    xt, yt = load_split(tmp_path, "cam", "test")
    assert len(yt) == 20 and (x[h == 2].shape[0] == 20)
    xtr, ytr = load_split(tmp_path, "cam", "train")
    xv, yv = load_split(tmp_path, "cam", "val")
    assert len(ytr) + len(yv) == 80 and set(yv) == {0, 1}
