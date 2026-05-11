#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* ------------------------------------------------------------------ */
/*  Forward declaration of the single CUDA kernel launcher.            */
/* ------------------------------------------------------------------ */

extern "C" void conv_launch(const float* d_weight, const float* d_input,
                            float* d_output,
                            int Ny, int Nx, int Ky, int Kx, int Ni, int Nn,
                            int NXPAD, int NXSCL);

/* ------------------------------------------------------------------ */
/*  Compile-time tile sizes come from -D flags via the Makefile.       */
/*  Mirror conv.cu's defaults here so unflagged builds match.          */
/* ------------------------------------------------------------------ */

#ifndef TILE_X
#define TILE_X 8
#endif
#ifndef TILE_Y
#define TILE_Y 8
#endif

/* ------------------------------------------------------------------ */
/*  Single source of truth for defaults (Conv1: 224^2, 3x3, 64/64).   */
/*  ny/nx: fixed problem shape (not CLI). ni/nn/ky/kx: CLI defaults.    */
/*  To add a new default: append a field + one line in print_defaults.   */
/* ------------------------------------------------------------------ */

struct Defaults {
    int ny     = 224;   // Conv1 output height — fixed, not on CLI
    int nx     = 224;   // Conv1 output width  — fixed, not on CLI
    int ky     = 3;
    int kx     = 3;
    int ni     = 64;
    int nn     = 64;
    int tile_x = 8;
    int tile_y = 8;
};
static constexpr Defaults kDefaults = {};

// CHUNK_NI is compile-time in conv.cu / Makefile, fixed at 64 for this harness.
static constexpr int kChunkNi = 64;

struct RunConfig {
    int ni;
    int nn;
    int ky;
    int kx;
    int tile_x;
    int tile_y;
};

/* ------------------------------------------------------------------ */
/*  Local indexing helpers (used by the CPU reference).                */
/* ------------------------------------------------------------------ */

static inline int idx3(int a, int b, int c, int B, int C) {
    return a * B * C + b * C + c;
}

static inline int idx4(int a, int b, int c, int d, int B, int C, int D) {
    return a * B * C * D + b * C * D + c * D + d;
}

/* ------------------------------------------------------------------ */
/*  CPU reference for correctness                                      */
/* ------------------------------------------------------------------ */

static void conv2d_cpu(const float* weight, const float* input, float* output,
                       int Ny, int Nx, int Ky, int Kx, int Ni, int Nn,
                       int NXPAD, int NXSCL) {
    for (int y = 0; y < Ny; y++)
        for (int x = 0; x < Nx; x++)
            for (int n = 0; n < Nn; n++) {
                float acc = 0.0f;
                for (int ky = 0; ky < Ky; ky++)
                    for (int kx = 0; kx < Kx; kx++)
                        for (int ni = 0; ni < Ni; ni++)
                            acc += weight[idx4(ky, kx, n, ni, Kx, Nn, Ni)]
                                 * input[idx3(y + ky, x + kx, ni, NXPAD, Ni)];
                output[idx3(y, x, n, NXSCL, Nn)] = acc;
            }
}

static bool check_results(const float* gpu, const float* cpu, int count,
                          int NXSCL, int Nn) {
    float max_diff = 0.0f;
    int   worst    = -1;

    for (int i = 0; i < count; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            worst    = i;
        }
    }

    if (max_diff < 1e-3f) {
        printf("  PASS  (max |diff| = %e)\n", max_diff);
        return true;
    }

    int n = worst % Nn;
    int x = (worst / Nn) % NXSCL;
    int y = worst / (NXSCL * Nn);
    printf("  FAIL  at output[%d][%d][%d]: GPU=%e  CPU=%e  |diff|=%e\n",
           y, x, n, gpu[worst], cpu[worst], max_diff);
    return false;
}

/* ------------------------------------------------------------------ */
/*  One convolution run: alloc, copy, launch, check, free.             */
/*  CPU correctness check is unconditional by design.                  */
/* ------------------------------------------------------------------ */

static void run_conv(int Ny, int Nx, int Ky, int Kx, int Ni, int Nn) {
    int NYPAD = Ny + Ky - 1;
    int NXPAD = Nx + Kx - 1;
    int NXSCL = Nx;

    size_t weight_count = (size_t)Ky * Kx * Nn * Ni;
    size_t input_count  = (size_t)NYPAD * NXPAD * Ni;
    size_t output_count = (size_t)Ny * NXSCL * Nn;

    size_t weight_bytes = weight_count * sizeof(float);
    size_t input_bytes  = input_count  * sizeof(float);
    size_t output_bytes = output_count * sizeof(float);

    printf("\n=== Ny=%d Nx=%d Ky=%d Kx=%d Ni=%d Nn=%d (tile=%dx%d) ===\n",
           Ny, Nx, Ky, Kx, Ni, Nn, TILE_X, TILE_Y);

    float* h_weight = (float*)malloc(weight_bytes);
    float* h_input  = (float*)malloc(input_bytes);
    float* h_output = (float*)malloc(output_bytes);
    float* h_ref    = (float*)malloc(output_bytes);

    srand(42);
    for (size_t i = 0; i < weight_count; i++)
        h_weight[i] = (float)rand() / (float)RAND_MAX * 0.01f;
    for (size_t i = 0; i < input_count; i++)
        h_input[i]  = (float)rand() / (float)RAND_MAX * 0.01f;
    memset(h_output, 0, output_bytes);
    memset(h_ref,    0, output_bytes);

    float *d_weight, *d_input, *d_output;
    cudaMalloc(&d_weight, weight_bytes);
    cudaMalloc(&d_input,  input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemcpy(d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input,  h_input,  input_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, output_bytes, cudaMemcpyHostToDevice);

    /* Launch + sync. Kernel time is measured externally by Nsight Compute. */
    conv_launch(d_weight, d_input, d_output,
                Ny, Nx, Ky, Kx, Ni, Nn, NXPAD, NXSCL);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "  Kernel error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    printf("  Running CPU reference...\n");
    conv2d_cpu(h_weight, h_input, h_ref,
               Ny, Nx, Ky, Kx, Ni, Nn, NXPAD, NXSCL);
    check_results(h_output, h_ref, (int)output_count, NXSCL, Nn);

cleanup:
    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_weight);
    free(h_input);
    free(h_output);
    free(h_ref);
}

/* ------------------------------------------------------------------ */
/*  CLI                                                                */
/* ------------------------------------------------------------------ */

static void print_help(FILE* out, const char* prog) {
    fprintf(out,
            "Usage:\n"
            "  %s [--ni NI] [--nn NN] [--ky K] [--kx K]\n"
            "  %*s [--tile-x TX] [--tile-y TY]\n"
            "  %*s [--print-defaults] [--help]\n"
            "\n"
            "Runs one Conv1-shaped convolution (Ny=Nx=224 fixed in this harness)\n"
            "and verifies against a CPU reference. Kernel time is measured\n"
            "externally by Nsight Compute.\n"
            "\n"
            "Tunable knobs (see --print-defaults for defaults):\n"
            "  --ni NI          input channels  (default %d)\n"
            "  --nn NN          output channels (default %d)\n"
            "  --ky K           filter height    (default %d)\n"
            "  --kx K           filter width     (default %d)\n"
            "  --tile-x TX      must equal compile-time TILE_X (%d). Default %d\n"
            "  --tile-y TY      must equal compile-time TILE_Y (%d). Default %d\n"
            "  --print-defaults print defaults JSON (ny/nx are fixed Conv1 shape)\n"
            "  --help, -h       show this message\n"
            "\n"
            "Fixed: Ny=%d Nx=%d (not CLI), CHUNK_NI=%d (Makefile / conv.cu).\n"
            "GPU kernel assumes Ni and per-chunk sizes are multiples of 4\n"
            "(float4 loads).\n",
            prog, (int)strlen(prog), "",
            (int)strlen(prog), "",
            kDefaults.ni, kDefaults.nn,
            kDefaults.ky, kDefaults.kx,
            TILE_X, kDefaults.tile_x,
            TILE_Y, kDefaults.tile_y,
            kDefaults.ny, kDefaults.nx, kChunkNi);
}

/* Emit defaults as a tiny JSON object. Single source of truth for       */
/* cross-language consumers (see model/defaults.py).                     */
static void print_defaults_json(FILE* out) {
    fprintf(out,
            "{\"ny\":%d,\"nx\":%d,\"ky\":%d,\"kx\":%d,"
            "\"ni\":%d,\"nn\":%d,\"chunk_ni\":%d,"
            "\"tile_x\":%d,\"tile_y\":%d}\n",
            kDefaults.ny, kDefaults.nx,
            kDefaults.ky, kDefaults.kx,
            kDefaults.ni, kDefaults.nn, kChunkNi,
            kDefaults.tile_x, kDefaults.tile_y);
}

static int parse_int_arg(const char* flag, const char* value, int* out) {
    if (value == nullptr) {
        fprintf(stderr, "error: %s requires an integer argument\n", flag);
        return -1;
    }
    char* end = nullptr;
    long v = strtol(value, &end, 10);
    if (end == value || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "error: %s requires a positive integer (got '%s')\n",
                flag, value);
        return -1;
    }
    *out = (int)v;
    return 0;
}

/* Returns 0 on success, 1 if a "stop after parsing" flag was handled    */
/* (--help, --print-defaults), -1 on parse error.                        */
static int parse_args(int argc, char** argv, RunConfig* cfg) {
    cfg->ni     = kDefaults.ni;
    cfg->nn     = kDefaults.nn;
    cfg->ky     = kDefaults.ky;
    cfg->kx     = kDefaults.kx;
    cfg->tile_x = kDefaults.tile_x;
    cfg->tile_y = kDefaults.tile_y;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            print_help(stdout, argv[0]);
            return 1;
        }
        if (strcmp(a, "--print-defaults") == 0) {
            print_defaults_json(stdout);
            return 1;
        }
        if (strcmp(a, "--ni") == 0) {
            const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
            if (parse_int_arg("--ni", v, &cfg->ni) != 0) return -1;
            continue;
        }
        if (strcmp(a, "--nn") == 0) {
            const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
            if (parse_int_arg("--nn", v, &cfg->nn) != 0) return -1;
            continue;
        }
        if (strcmp(a, "--ky") == 0) {
            const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
            if (parse_int_arg("--ky", v, &cfg->ky) != 0) return -1;
            continue;
        }
        if (strcmp(a, "--kx") == 0) {
            const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
            if (parse_int_arg("--kx", v, &cfg->kx) != 0) return -1;
            continue;
        }
        if (strcmp(a, "--tile-x") == 0) {
            const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
            if (parse_int_arg("--tile-x", v, &cfg->tile_x) != 0) return -1;
            continue;
        }
        if (strcmp(a, "--tile-y") == 0) {
            const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
            if (parse_int_arg("--tile-y", v, &cfg->tile_y) != 0) return -1;
            continue;
        }

        fprintf(stderr, "error: unknown argument '%s'\n\n", a);
        print_help(stderr, argv[0]);
        return -1;
    }
    return 0;
}

static int validate_tile_matches_compile(int tile_x, int tile_y) {
    if (tile_x == TILE_X && tile_y == TILE_Y) return 0;
    fprintf(stderr,
            "error: --tile-x %d --tile-y %d does not match this binary's "
            "compile-time TILE_X=%d / TILE_Y=%d.\n"
            "       Rebuild with `make tiled TILE_X=%d TILE_Y=%d CHUNK_NI=%d` "
            "or invoke the matching bin/conv_t<TX>_<TY>_c<C> binary.\n",
            tile_x, tile_y, TILE_X, TILE_Y, tile_x, tile_y, kChunkNi);
    return -1;
}

int main(int argc, char** argv) {
    RunConfig cfg;
    int rc = parse_args(argc, argv, &cfg);
    if (rc == 1) return 0;
    if (rc < 0)  return 1;

    if (validate_tile_matches_compile(cfg.tile_x, cfg.tile_y) != 0) return 1;

    run_conv(kDefaults.ny, kDefaults.nx,
             cfg.ky, cfg.kx, cfg.ni, cfg.nn);
    return 0;
}
