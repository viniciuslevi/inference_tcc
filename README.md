# inference_tcc

Modelos CNN INT8 do TCC **"Aceleração de Redes Neurais em FPGA Cyclone IV E"**
(Vinícius Levi dos Santos, Engenharia da Computação, UNIVASF, 2026): treinamento,
quantização pós-treinamento (PTQ) e referência bit-exata do *firmware* que executa a
inferência no processador NEORV32 com o acelerador sistólico, na placa DE2-115.

Faz parte de três repositórios:

| Repositório | Papel |
|---|---|
| [`neorv32`](https://github.com/viniciuslevi/neorv32/tree/systolic-array-accelerator) (ramo `systolic-array-accelerator`) | Hardware (acelerador, controlador SDRAM, SoC) e *firmware* `sw/tcc/mnist_infer` |
| `inference_tcc` (este) | Treinamento, quantização INT8 e referência bit-exata dos modelos |
| [`neorv32_infer_desktop`](https://github.com/viniciuslevi/neorv32_infer_desktop) | Aplicativo de operação da placa e coleta em lote |

## Modelos

Arquitetura: conv3×3(1→32)+ReLU → maxpool → conv3×3(32→64)+ReLU → maxpool →
dense(3136→128)+ReLU → dense(128→10). Treino em PyTorch (Adam, lr 1e-3, lote 128,
semente 0; 3 épocas no MNIST e 10 no Fashion-MNIST). Quantização: pesos INT8
simétricos com uma escala por camada, *bias* INT32, ativações INT8 assimétricas
calibradas nas 2.000 primeiras imagens de treino, requantização em float32 com
arredondamento *half away from zero*, como no firmware.

Acurácia no conjunto de teste completo (10.000 imagens), com `avalia.py`:

| Dataset | FP32 | INT8 (simulação bit-exata do firmware) | INT8, 2.000 primeiras imagens |
|---|---|---|---|
| MNIST | 98,72% | 98,72% | 98,20% |
| Fashion-MNIST | 91,78% | 91,96% | 91,85% |

Na placa, as 200 primeiras imagens de teste de cada conjunto deram logits idênticos
aos desta simulação em 400 de 400 casos.

Os dois modelos usam a mesma arquitetura e o mesmo layout de `weights.bin`
(422.344 B); só muda o `model_meta.h` (escalas e zero-points). Os arquivos em `out/`
são exatamente os usados nas medições do TCC.

## Conteúdo de `out/<dataset>/`

| Arquivo | O que é |
|---|---|
| `weights.bin` | pesos INT8 (layout K×N) + bias INT32 |
| `model_meta.h` | substitui o `model_meta.h` do firmware |
| `images/*.bin` | 10 imagens de teste (uma por classe), 784 B INT8 HWC (`pixel - 128`) |
| `expected.txt` | classe e logits INT8 que a placa deve imprimir |
| `golden/<img>/*.bin` | ativações INT8 esperadas por estágio (conv1, pool1, conv2, pool2, dense1, logits) |
| `<dataset>_float.pt` | modelo float (PyTorch) |

As saídas esperadas vêm de uma simulação em NumPy bit-exata de `inference.c` +
`accel_driver.c` (requantização em float32, correção de zero-point). As imagens
de teste são a primeira de cada classe que o modelo INT8 acerta.

## Uso

Testado com Python 3.12.7, PyTorch 2.14.0 (CPU) e NumPy 2.5.3.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy pyserial
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# treinar, quantizar e exportar (baixa os datasets em data/)
.venv/Scripts/python build_mnist_int8.py --dataset mnist
.venv/Scripts/python build_mnist_int8.py --dataset fashion_mnist

# acurácia FP32 e INT8 no teste completo, com os arquivos de out/
.venv/Scripts/python avalia.py
```

Retreinar sobrescreve `out/`; com outra versão do PyTorch, os pesos podem diferir
dos usados no TCC.

Na placa (ou use o aplicativo `neorv32_infer_desktop`, que empacota firmware e pesos):

```bash
cp out/fashion_mnist/model_meta.h <neorv32>/sw/tcc/mnist_infer/model_meta.h
# recompilar (make clean exe) e subir o firmware
python <neorv32>/sw/tcc/mnist_infer/tools/send_weights.py --port COM17 \
    --weights out/fashion_mnist/weights.bin \
    --image   out/fashion_mnist/images/img_06_shirt.bin
```
