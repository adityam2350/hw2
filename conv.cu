#include <cuda_runtime.h>

static __host__ __device__ inline int idx3(int a, int b, int c,
                                           int B, int C) {
    return a * B * C + b * C + c;
}

static __host__ __device__ inline int idx4(int a, int b, int c, int d,
                                           int B, int C, int D) {
    return a * B * C * D + b * C * D + c * D + d;
}

#ifndef TILE_Y
#define TILE_Y 8
#endif
#ifndef TILE_X
#define TILE_X 8
#endif
#ifndef CHUNK_NI
#define CHUNK_NI 64
#endif
#ifndef MAX_N_PER_THREAD
#define MAX_N_PER_THREAD 2
#endif

__global__ void conv_kernel(const float* __restrict__ weight,
                            const float* __restrict__ input,
                            float*       __restrict__ output,
                            int Ny, int Nx,
                            int Ky, int Kx, int Ni, int Nn,
                            int NXPAD, int NXSCL) {
    int bx = blockIdx.x * TILE_X;
    int by = blockIdx.y * TILE_Y;

    extern __shared__ float s_input[];

    int NYPAD       = Ny + Ky - 1;
    int inp_tile_x  = TILE_X + Kx - 1;
    int inp_tile_y  = TILE_Y + Ky - 1;

    int n_count = (Nn + blockDim.x - 1) / blockDim.x;
    if (n_count > MAX_N_PER_THREAD) n_count = MAX_N_PER_THREAD;

    float acc[MAX_N_PER_THREAD][TILE_Y][TILE_X];
    for (int j = 0; j < MAX_N_PER_THREAD; j++)
        for (int ty = 0; ty < TILE_Y; ty++)
            for (int tx = 0; tx < TILE_X; tx++)
                acc[j][ty][tx] = 0.0f;

    for (int ni_base = 0; ni_base < Ni; ni_base += CHUNK_NI) {
        int chunk = Ni - ni_base;
        if (chunk > CHUNK_NI) chunk = CHUNK_NI;

        int tile_elems = inp_tile_y * inp_tile_x * chunk;
        int vec_count  = tile_elems / 4;

        for (int vi = threadIdx.x; vi < vec_count; vi += blockDim.x) {
            int base = vi * 4;
            int c    = base % chunk;
            int s    = base / chunk;
            int ix   = s % inp_tile_x;
            int iy   = s / inp_tile_x;
            int gy   = by + iy;
            int gx   = bx + ix;

            float4 v;
            if (gy < NYPAD && gx < NXPAD) {
                const float* src =
                    &input[idx3(gy, gx, ni_base + c, NXPAD, Ni)];
                v = *reinterpret_cast<const float4*>(src);
            } else {
                v = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            *reinterpret_cast<float4*>(&s_input[base]) = v;
        }
        __syncthreads();

        for (int j = 0; j < n_count; j++) {
            int n = threadIdx.x + j * blockDim.x;
            if (n >= Nn) continue;

            for (int fky = 0; fky < Ky; fky++) {
                for (int fkx = 0; fkx < Kx; fkx++) {
                    for (int c = 0; c < chunk; c += 4) {
                        const float4 wv = *reinterpret_cast<const float4*>(
                            &weight[idx4(fky, fkx, n, ni_base + c, Kx, Nn, Ni)]);

                        for (int ty = 0; ty < TILE_Y; ty++) {
                            for (int tx = 0; tx < TILE_X; tx++) {
                                const float4 iv = *reinterpret_cast<const float4*>(
                                    &s_input[idx3(ty + fky, tx + fkx, c,
                                                  inp_tile_x, chunk)]);
                                acc[j][ty][tx] += wv.x * iv.x + wv.y * iv.y
                                                + wv.z * iv.z + wv.w * iv.w;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < n_count; j++) {
        int n = threadIdx.x + j * blockDim.x;
        if (n >= Nn) continue;
        for (int ty = 0; ty < TILE_Y; ty++) {
            for (int tx = 0; tx < TILE_X; tx++) {
                int oy = by + ty;
                int ox = bx + tx;
                if (oy < Ny && ox < Nx)
                    output[idx3(oy, ox, n, NXSCL, Nn)] = acc[j][ty][tx];
            }
        }
    }
}

extern "C" void conv_launch(const float* d_weight, const float* d_input,
                            float* d_output,
                            int Ny, int Nx, int Ky, int Kx, int Ni, int Nn,
                            int NXPAD, int NXSCL) {
    int threads = (Nn < 256) ? Nn : 256;
    dim3 block(threads);
    dim3 grid((Nx + TILE_X - 1) / TILE_X, (Ny + TILE_Y - 1) / TILE_Y);

    int inp_tile_y = TILE_Y + Ky - 1;
    int inp_tile_x = TILE_X + Kx - 1;
    int chunk = (Ni < CHUNK_NI) ? Ni : CHUNK_NI;
    int shmem = inp_tile_y * inp_tile_x * chunk * (int)sizeof(float);

    conv_kernel<<<grid, block, shmem>>>(d_weight, d_input, d_output,
                                        Ny, Nx, Ky, Kx, Ni, Nn,
                                        NXPAD, NXSCL);
}
