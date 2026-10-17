"""Regenerate tests/fixtures/mini_crc.parquet: 27 synthetic 224x224 PNG tiles, 3 per class."""
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def main(out):
    rng = np.random.default_rng(0)
    imgs, labels = [], []
    for c in range(9):
        for _ in range(3):
            a = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            a[..., 0] = c * 25  # class-coded red channel (survives PNG exactly)
            b = io.BytesIO()
            Image.fromarray(a).save(b, format="PNG")
            imgs.append({"bytes": b.getvalue(), "path": f"c{c}.png"})
            labels.append(c)
    t = pa.table({"image": pa.array(imgs, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                  "label": pa.array(labels, type=pa.int64())})
    pq.write_table(t, out, row_group_size=10)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
