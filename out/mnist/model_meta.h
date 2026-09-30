// Gerado automaticamente por build_mnist_int8.py -- NAO EDITAR A MAO.
// Fase 3 da metodologia do TCC: offsets/dimensoes/quantizacao do modelo
// CNN mnist (INT8) para carga na SDRAM e inferencia via acelerador sistolico.
#ifndef MODEL_META_H
#define MODEL_META_H

#include <stdint.h>

#define MODEL_WEIGHTS_BYTES 422344u

typedef struct {
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
} model_layer_t;

#define MODEL_NUM_LAYERS 4

static const model_layer_t MODEL_LAYERS[MODEL_NUM_LAYERS] = {
    { "conv1", 1, 9u, 32u, 28u, 28u, 1u, 0u, 288u, 288u, 128u, 0.00392156886f, -128, 0.00472341152f, 0.00589119876f, -128, 1 },
    { "conv2", 1, 288u, 64u, 14u, 14u, 32u, 416u, 18432u, 18848u, 256u, 0.00589119876f, -128, 0.00290589617f, 0.0223924667f, -128, 1 },
    { "dense1", 0, 3136u, 128u, 0u, 0u, 0u, 19104u, 401408u, 420512u, 512u, 0.0223924667f, -128, 0.00213918765f, 0.109570488f, -128, 1 },
    { "dense2", 0, 128u, 10u, 0u, 0u, 0u, 421024u, 1280u, 422304u, 40u, 0.109570488f, -128, 0.00162068324f, 0.18547374f, 6, 0 },
};

#define LAYER_CONV1 0
#define LAYER_CONV2 1
#define LAYER_DENSE1 2
#define LAYER_DENSE2 3

#define INPUT_H 28
#define INPUT_W 28
#define INPUT_C 1
#define INPUT_SCALE 0.00392156886f
#define INPUT_ZERO_POINT -128

#define OUTPUT_CLASSES 10
#define OUTPUT_SCALE 0.18547374f
#define OUTPUT_ZERO_POINT 6

#endif // MODEL_META_H
