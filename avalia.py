#!/usr/bin/env python3
"""
Acuracia no computador hospedeiro (Tabela de acuracia no host do TCC).

Para cada conjunto (MNIST e Fashion-MNIST), avalia sobre o conjunto de teste:
  - o modelo FP32 treinado (out/<ds>/<ds>_float.pt);
  - a simulacao bit-exata do firmware (INT8), a partir do weights.bin e do
    model_meta.h exportados -- os mesmos arquivos gravados na placa.

Reusa a arquitetura e a simulacao de build_mnist_int8.py, sem retreinar:

    .venv/Scripts/python avalia.py
"""
import argparse
import os
import re
import sys

import numpy as np

RE_ROW = re.compile(
    r'\{\s*"(\w+)",\s*(\d+),\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*'
    r'(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*'
    r'([-\d.e+]+)f,\s*(-?\d+),\s*([-\d.e+]+)f,\s*([-\d.e+]+)f,\s*(-?\d+),\s*(\d+)\s*\}')


def load_layers(out_dir):
    """Le as camadas quantizadas do model_meta.h + weights.bin exportados."""
    f32 = np.float32
    with open(os.path.join(out_dir, "model_meta.h"), encoding="utf-8") as f:
        rows = RE_ROW.findall(f.read())
    blob = open(os.path.join(out_dir, "weights.bin"), "rb").read()
    layers = []
    for r in rows:
        k, n = int(r[2]), int(r[3])
        w_off, w_bytes, b_off, b_bytes = map(int, r[7:11])
        layers.append(dict(
            name=r[0], k=k, n=n,
            wq=np.frombuffer(blob, np.int8, w_bytes, w_off).reshape(k, n),
            bq=np.frombuffer(blob, "<i4", b_bytes // 4, b_off),
            in_scale=f32(r[11]), in_zp=int(r[12]), w_scale=f32(r[13]),
            out_scale=f32(r[14]), out_zp=int(r[15]), relu=int(r[16])))
    assert len(layers) == 4, f"{out_dir}: esperava 4 camadas"
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inference-tcc", default=os.path.dirname(os.path.abspath(__file__)),
                    help="pasta do repositorio inference_tcc (padrao: a deste script)")
    args = ap.parse_args()
    root = os.path.abspath(args.inference_tcc)
    sys.path.insert(0, root)
    import torch
    import build_mnist_int8 as b

    print("# Acuracia no hospedeiro: FP32 e INT8 (simulacao bit-exata do firmware)")
    print(f"# Python {sys.version.split()[0]}, PyTorch {torch.__version__}, NumPy {np.__version__}")
    for ds in ("mnist", "fashion_mnist"):
        b.DATA_DIR = os.path.join(root, "data", ds)
        out_dir = os.path.join(root, "out", ds)
        d = b.load_idx_dataset(b.DATASETS[ds]["url"])
        x, y = d["test_x"], d["test_y"]

        model = b.Net()
        model.load_state_dict(torch.load(os.path.join(out_dir, f"{ds}_float.pt"), map_location="cpu"))
        acc_f = b.float_accuracy(model, x, y)

        layers = load_layers(out_dir)
        pred = np.array([b.argmax_first(b.sim_firmware(b.img_to_i8(x[i]), layers)["logits"])
                         for i in range(len(x))])
        ok = pred == y
        n = len(y)
        print(f"\n[{ds}] conjunto de teste: {n} imagens")
        print(f"  FP32:                       {int(round(acc_f * n))} / {n} = {acc_f * 100:.2f} %")
        print(f"  INT8 (sim. firmware):       {ok.sum()} / {n} = {ok.mean() * 100:.2f} %")
        print(f"  INT8, 2.000 primeiras:      {ok[:2000].sum()} / 2000 = {ok[:2000].mean() * 100:.2f} %")
        print(f"  INT8, 200 primeiras (0-199): {ok[:200].sum()} / 200 = {ok[:200].mean() * 100:.2f} %")


if __name__ == "__main__":
    sys.exit(main())
