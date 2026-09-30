# inference_tcc

Modelos CNN INT8 para o firmware `neorv32_mnist_infer` (NEORV32 + acelerador
sistólico 8×8 na DE2-115). Gera os binários enviados à FPGA por
`neorv32/sw/example/neorv32_mnist_infer/tools/send_weights.py`.

## Datasets

| Dataset | Acurácia float | Acurácia INT8 (sim. firmware) | Saídas |
|---|---|---|---|
| MNIST | 98,72% | 98,20% | [`out/mnist/`](out/mnist/) |
| Fashion-MNIST | 91,78% | 91,85% | [`out/fashion_mnist/`](out/fashion_mnist/) |

Os dois usam a mesma arquitetura e o mesmo layout de `weights.bin` (422 344 B);
só muda o `model_meta.h` (escalas/zero-points). Nenhuma mudança de código no firmware.

Arquitetura: conv3×3(1→32)+ReLU → maxpool → conv3×3(32→64)+ReLU → maxpool →
dense(3136→128)+ReLU → dense(128→10).

## Conteúdo de `out/<dataset>/`

| Arquivo | O que é |
|---|---|
| `weights.bin` | pesos INT8 (layout K×N) + bias INT32 |
| `model_meta.h` | substitui o `model_meta.h` do firmware |
| `images/*.bin` | 10 imagens de teste (uma por classe), 784 B INT8 HWC (`pixel - 128`) |
| `expected.txt` | classe e logits INT8 que a placa deve imprimir |
| `golden/<img>/*.bin` | ativações INT8 esperadas por estágio (conv1, pool1, conv2, pool2, dense1, logits) |
| `<dataset>_float.pt` | modelo float (PyTorch) |

As saídas esperadas vêm de uma simulação em numpy bit-exata de `inference.c` +
`accel_driver.c` (requantização em float32, correção de zero-point). As imagens
de teste são a primeira de cada classe que o modelo INT8 acerta.

## Uso

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy pyserial
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python build_mnist_int8.py --dataset mnist
.venv/Scripts/python build_mnist_int8.py --dataset fashion_mnist
```

Na placa:

```bash
cp out/fashion_mnist/model_meta.h <neorv32>/sw/example/neorv32_mnist_infer/model_meta.h
# recompilar (make clean elf exe) e subir o firmware
python tools/send_weights.py --port COM17 \
    --weights <inference_tcc>/out/fashion_mnist/weights.bin \
    --image   <inference_tcc>/out/fashion_mnist/images/img_06_shirt.bin
```
