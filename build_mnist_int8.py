#!/usr/bin/env python3
"""
build_mnist_int8.py
-------------------
Treina a CNN esperada pelo firmware neorv32_mnist_infer em um dataset 28x28x1 de
10 classes (MNIST ou Fashion-MNIST), quantiza para INT8 e gera os binarios que
tools/send_weights.py envia para a FPGA. Ambos os datasets usam exatamente a mesma
arquitetura e o mesmo layout de weights.bin; so muda o model_meta.h (escalas).

Uso:
    python build_mnist_int8.py --dataset mnist
    python build_mnist_int8.py --dataset fashion_mnist

Arquitetura (identica a inference.c / model_meta.h):
    conv1 3x3 same, 1->32, ReLU -> maxpool 2x2
    conv2 3x3 same, 32->64, ReLU -> maxpool 2x2
    flatten (ordem HWC) -> dense1 3136->128, ReLU -> dense2 128->10

Quantizacao (mesma semantica do firmware / spec INT8 do TFLite):
    - pesos: INT8 simetrico por tensor (zp=0), layout (K, N) row-major
      conv: K = (dy, dx, cin), dy mais externo
    - bias: INT32 = round(b / (in_scale * w_scale))
    - ativacoes: INT8 assimetrico; entrada scale=1/255, zp=-128
    - requantizacao em float32, arredondamento "half away from zero", como requantize()

Saidas (em ./out/<dataset>):
    weights.bin            pesos INT8 + bias INT32 de todas as camadas
    model_meta.h           substitui o model_meta.h do firmware
    images/img_NN_labelL.bin   imagens de teste 28x28 INT8 (784 B, HWC)
    golden/img_NN/*.bin    ativacoes INT8 esperadas de cada camada (simulacao bit-exata)
    expected.txt           label real, logits e classe esperados por imagem
"""
import argparse
import gzip
import os
import struct
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "mnist": dict(
        url="https://ossci-datasets.s3.amazonaws.com/mnist/",
        epochs=3,
        classes=[str(i) for i in range(10)],
    ),
    "fashion_mnist": dict(
        url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        epochs=10,
        classes=["tshirt", "trouser", "pullover", "dress", "coat",
                 "sandal", "shirt", "sneaker", "bag", "ankle_boot"],
    ),
}
DATA_DIR = OUT_DIR = None  # definidos em main() a partir de --dataset
IDX_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
SEED = 0


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def load_idx_dataset(url):
    os.makedirs(DATA_DIR, exist_ok=True)
    arrays = {}
    for key, fname in IDX_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[*] baixando {fname}")
            urllib.request.urlretrieve(url + fname, path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if key.endswith("_x"):
            n, h, w = struct.unpack(">III", raw[4:16])
            arrays[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(n, h, w)
        else:
            arrays[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    return arrays


# ---------------------------------------------------------------------------
# Modelo float
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dense1 = nn.Linear(7 * 7 * 64, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x, taps=None):
        x = F.relu(self.conv1(x));  t1 = x
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x));  t2 = x
        x = F.max_pool2d(x, 2)
        x = x.permute(0, 2, 3, 1).flatten(1)  # HWC, igual ao firmware
        x = F.relu(self.dense1(x)); t3 = x
        x = self.dense2(x);         t4 = x
        if taps is not None:
            taps.extend([t1, t2, t3, t4])
        return x


def to_tensor(x_u8):
    return torch.from_numpy(x_u8.astype(np.float32) / 255.0).unsqueeze(1)


def train(model, x, y, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        model.train()
        tot = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = to_tensor(x[idx]), torch.from_numpy(y[idx])
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"[train] epoca {ep + 1}/{epochs}: loss={tot / n:.4f}")


@torch.no_grad()
def float_accuracy(model, x, y):
    model.eval()
    pred = np.concatenate([model(to_tensor(x[i:i + 1000])).argmax(1).numpy()
                           for i in range(0, len(x), 1000)])
    return (pred == y).mean()


# ---------------------------------------------------------------------------
# Quantizacao
# ---------------------------------------------------------------------------
f32 = np.float32


def weights_kn(model):
    """Pesos/bias float no layout (K, N) do acelerador."""
    c1 = model.conv1.weight.detach().numpy()  # (Cout, Cin, 3, 3)
    c2 = model.conv2.weight.detach().numpy()
    d1 = model.dense1.weight.detach().numpy()  # (N, K), K ja em ordem HWC
    d2 = model.dense2.weight.detach().numpy()
    conv_kn = lambda w: w.transpose(2, 3, 1, 0).reshape(-1, w.shape[0])  # (dy,dx,cin) x cout
    return [
        (conv_kn(c1), model.conv1.bias.detach().numpy()),
        (conv_kn(c2), model.conv2.bias.detach().numpy()),
        (d1.T.copy(), model.dense1.bias.detach().numpy()),
        (d2.T.copy(), model.dense2.bias.detach().numpy()),
    ]


@torch.no_grad()
def calibrate(model, x_calib):
    """Faixas (min, max) das saidas de cada camada (pre-pool, pos-ReLU)."""
    model.eval()
    taps = []
    model(to_tensor(x_calib), taps)
    return [(float(t.min()), float(t.max())) for t in taps]


def act_qparams(lo, hi):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / 255.0
    zp = int(np.clip(round(-128 - lo / scale), -128, 127))
    return f32(scale), zp


def quantize(model, ranges):
    names = ["conv1", "conv2", "dense1", "dense2"]
    geo = [(1, 28, 28, 1), (1, 14, 14, 32), (0, 0, 0, 0), (0, 0, 0, 0)]
    relu = [1, 1, 1, 0]
    in_scale, in_zp = f32(1.0 / 255.0), -128
    layers = []
    for i, (w, b) in enumerate(weights_kn(model)):
        w_scale = f32(np.abs(w).max() / 127.0)
        wq = np.clip(np.round(w / w_scale), -127, 127).astype(np.int8)
        bq = np.round(b / (float(in_scale) * float(w_scale))).astype(np.int32)
        if relu[i]:
            out_scale, out_zp = f32(ranges[i][1] / 255.0), -128
        else:
            out_scale, out_zp = act_qparams(*ranges[i])
        is_conv, h, w_, c = geo[i]
        layers.append(dict(name=names[i], is_conv=is_conv, k=wq.shape[0], n=wq.shape[1],
                           in_h=h, in_w=w_, in_c=c, wq=wq, bq=bq,
                           in_scale=in_scale, in_zp=in_zp, w_scale=w_scale,
                           out_scale=out_scale, out_zp=out_zp, relu=relu[i]))
        in_scale, in_zp = out_scale, out_zp  # maxpool preserva escala/zp
    return layers


# ---------------------------------------------------------------------------
# Simulacao bit-exata do firmware (inference.c + accel_driver.c)
# ---------------------------------------------------------------------------
def im2col(x_hwc, zp):
    H, W, C = x_hwc.shape
    p = np.pad(x_hwc, ((1, 1), (1, 1), (0, 0)), constant_values=zp)
    cols = [p[dy:dy + H, dx:dx + W, :] for dy in range(3) for dx in range(3)]
    return np.concatenate(cols, axis=2).reshape(H * W, 9 * C)


def requantize(acc, L):
    # mult = in_scale * w_scale / out_scale, tudo em float32 como no C
    mult = f32(f32(L["in_scale"] * L["w_scale"]) / L["out_scale"])
    v = acc.astype(np.float32) * mult
    v = np.where(v >= 0, v + f32(0.5), v - f32(0.5)).astype(np.float32)
    q = np.trunc(v).astype(np.int64) + L["out_zp"]
    lo = L["out_zp"] if L["relu"] else -128
    return np.clip(q, lo, 127).astype(np.int8)


def matmul_layer(a_i8, L):
    a = a_i8.astype(np.int64) - L["in_zp"]
    acc = a @ L["wq"].astype(np.int64) + L["bq"]
    assert np.abs(acc).max() < 2**31, "overflow INT32 no acumulador"
    return requantize(acc, L)


def maxpool(x_hwc):
    H, W, C = x_hwc.shape
    return x_hwc.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def sim_firmware(img_i8, layers):
    x = img_i8.reshape(28, 28, 1)
    out = {}
    x = matmul_layer(im2col(x, layers[0]["in_zp"]), layers[0]).reshape(28, 28, 32); out["conv1"] = x
    x = maxpool(x);                                                               out["pool1"] = x
    x = matmul_layer(im2col(x, layers[1]["in_zp"]), layers[1]).reshape(14, 14, 64); out["conv2"] = x
    x = maxpool(x);                                                               out["pool2"] = x
    x = matmul_layer(x.reshape(1, -1), layers[2]).reshape(-1);                    out["dense1"] = x
    x = matmul_layer(x.reshape(1, -1), layers[3]).reshape(-1);                    out["logits"] = x
    return out


def argmax_first(v):
    return int(np.argmax(v))  # mesma regra do argmax_i8 (primeiro maximo)


def img_to_i8(img_u8):
    return (img_u8.reshape(-1).astype(np.int16) - 128).astype(np.int8)


# ---------------------------------------------------------------------------
# Exportacao
# ---------------------------------------------------------------------------
def fmt_f(x):
    return f"{float(x):.9g}f"  # 9 digitos: ida e volta exata em float32


def export(layers, dataset):
    blob = bytearray()
    for L in layers:
        L["w_offset"], L["w_bytes"] = len(blob), L["wq"].nbytes
        blob += L["wq"].tobytes()
        blob += b"\0" * (-len(blob) % 4)  # bias INT32 alinhado
        L["b_offset"], L["b_bytes"] = len(blob), L["bq"].nbytes
        blob += L["bq"].astype("<i4").tobytes()
    with open(os.path.join(OUT_DIR, "weights.bin"), "wb") as f:
        f.write(blob)

    rows = []
    for L in layers:
        rows.append(
            f'    {{ "{L["name"]}", {L["is_conv"]}, {L["k"]}u, {L["n"]}u, '
            f'{L["in_h"]}u, {L["in_w"]}u, {L["in_c"]}u, '
            f'{L["w_offset"]}u, {L["w_bytes"]}u, {L["b_offset"]}u, {L["b_bytes"]}u, '
            f'{fmt_f(L["in_scale"])}, {L["in_zp"]}, {fmt_f(L["w_scale"])}, '
            f'{fmt_f(L["out_scale"])}, {L["out_zp"]}, {L["relu"]} }},')
    last = layers[-1]
    header = f"""// Gerado automaticamente por build_mnist_int8.py -- NAO EDITAR A MAO.
// Fase 3 da metodologia do TCC: offsets/dimensoes/quantizacao do modelo
// CNN {dataset} (INT8) para carga na SDRAM e inferencia via acelerador sistolico.
#ifndef MODEL_META_H
#define MODEL_META_H

#include <stdint.h>

#define MODEL_WEIGHTS_BYTES {len(blob)}u

typedef struct {{
    const char *name;
    uint8_t  is_conv;      // 1 = conv2d (im2col), 0 = dense
    uint32_t k;            // dimensao de reducao (contracao)
    uint32_t n;            // canais/neuronios de saida
    uint32_t in_h, in_w, in_c;   // validos apenas se is_conv
    uint32_t w_offset, w_bytes;  // offset/tamanho dos pesos em weights.bin
    uint32_t b_offset, b_bytes;  // offset/tamanho do bias (INT32) em weights.bin
    float    in_scale;
    int32_t  in_zero_point;
    float    w_scale;
    float    out_scale;
    int32_t  out_zero_point;
    uint8_t  relu;
}} model_layer_t;

#define MODEL_NUM_LAYERS 4

static const model_layer_t MODEL_LAYERS[MODEL_NUM_LAYERS] = {{
{chr(10).join(rows)}
}};

#define LAYER_CONV1 0
#define LAYER_CONV2 1
#define LAYER_DENSE1 2
#define LAYER_DENSE2 3

#define INPUT_H 28
#define INPUT_W 28
#define INPUT_C 1
#define INPUT_SCALE {fmt_f(layers[0]["in_scale"])}
#define INPUT_ZERO_POINT {layers[0]["in_zp"]}

#define OUTPUT_CLASSES 10
#define OUTPUT_SCALE {fmt_f(last["out_scale"])}
#define OUTPUT_ZERO_POINT {last["out_zp"]}

#endif // MODEL_META_H
"""
    with open(os.path.join(OUT_DIR, "model_meta.h"), "w", newline="\n") as f:
        f.write(header)
    return len(blob)


def main():
    global DATA_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    DATA_DIR = os.path.join(HERE, "data", args.dataset)
    OUT_DIR = os.path.join(HERE, "out", args.dataset)
    classes = cfg["classes"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    d = load_idx_dataset(cfg["url"])

    model = Net()
    train(model, d["train_x"], d["train_y"], cfg["epochs"])
    acc_f = float_accuracy(model, d["test_x"], d["test_y"])
    print(f"[float] acuracia teste: {acc_f * 100:.2f}%")

    ranges = calibrate(model, d["train_x"][:2000])
    layers = quantize(model, ranges)

    # Acuracia INT8 (simulacao bit-exata do firmware) num subconjunto do teste
    n_eval = 2000
    correct = sum(argmax_first(sim_firmware(img_to_i8(d["test_x"][i]), layers)["logits"])
                  == d["test_y"][i] for i in range(n_eval))
    print(f"[int8 ] acuracia (sim. firmware, {n_eval} imagens de teste): {correct / n_eval * 100:.2f}%")

    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    nbytes = export(layers, args.dataset)
    print(f"[out] weights.bin: {nbytes} bytes; model_meta.h gerado")

    # Imagens de teste: primeira imagem de cada classe que o modelo INT8 acerta
    # (assim "classe prevista == label" vale como criterio de PASS na placa)
    lines = [f"# {args.dataset}: arquivo  idx_no_teste  label  classe_esperada  logits_int8 (saida esperada da placa)",
             "# classes: " + " ".join(f"{i}={c}" for i, c in enumerate(classes))]
    for digit in range(len(classes)):
        for idx in np.where(d["test_y"] == digit)[0]:
            img = img_to_i8(d["test_x"][idx])
            sim = sim_firmware(img, layers)
            if argmax_first(sim["logits"]) == digit:
                break
        idx = int(idx)
        tag = f"img_{digit:02d}_label{digit}" if args.dataset == "mnist" else f"img_{digit:02d}_{classes[digit]}"
        with open(os.path.join(OUT_DIR, "images", tag + ".bin"), "wb") as f:
            f.write(img.tobytes())
        gdir = os.path.join(OUT_DIR, "golden", tag)
        os.makedirs(gdir, exist_ok=True)
        for name, arr in sim.items():
            with open(os.path.join(gdir, name + ".bin"), "wb") as f:
                f.write(arr.astype(np.int8).tobytes())
        pred = argmax_first(sim["logits"])
        logits = " ".join(str(int(v)) for v in sim["logits"])
        lines.append(f"{tag}.bin  idx={idx:<5d} label={digit} pred={pred}  logits: {logits}")
        print(f"[img] {tag}.bin (idx {idx}) -> pred {pred}  logits [{logits}]")
    with open(os.path.join(OUT_DIR, "expected.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{args.dataset}_float.pt"))
    print(f"[ok] saidas em {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
