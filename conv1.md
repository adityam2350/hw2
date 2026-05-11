# Convolution \#1

## 1\. Parallelization strategy

Each block collaboratively loads a (10×10)×64 input patch into shared memory, where the 10×10 region comes from the 8×8 output tile plus the halo required for a K=3 kernel. This allows the same activations and weights to be reused across all 8×8 output pixels, significantly reducing redundant global memory accesses. Additionally, using float4 vector loads along the Ni dimension reduces the total number of global load instructions and improves memory throughput. However, if either dimension grows significantly—such as Nn exceeding the thread cap or Ni exceeding CHUNK\_NI—the kernel would require additional passes, larger shared-memory usage, and potentially a different mapping strategy to avoid excessive register pressure and reduced performance.

---

## 2\. Algorithmic FLOP count (derivation)

Each output value is a dot product over the **3×3** spatial window and all **64** input channels, so for **one** output you need **3 × 3 × 64 \= 576** multiply–adds (MACs). There is one output for each combination of height, width, and output channel, so the number of outputs is **224 × 224 × 64**.

Multiply those together to get the total MACs: **224 × 224 × 64 × 3 × 3 × 64 \= 1,849,688,064** MACs.

For FLOP counts, count **2 floating-point operations per MAC** (one multiply, one add), so the total is **2 × 1,849,688,064 \= 3,699,376,128** FLOPs. That is about **3.70 billion** FLOPs, or about **3700 GFLOPs**, for one forward pass of this layer.

---

## 3\. Execution time and achieved GFLOPS

| Kernel | Time (ms) | Achieved GFLOPS (3,699,376,128 FLOPs ÷ time) |
| :---- | :---- | :---- |
| `conv2d_naive` | 20.03 | 3,699,376,128 / 0.02003 ≈ **185** |
| `conv2d_conv1_optimized` | 1.21 | 3,699,376,128 / 0.00121 ≈ **3,060** |

*(GFLOPS uses the **same algorithmic FLOP count** for both, as required by the question.)*

---

## 4\. Roofline analysis

**Hardware (approximate, for TITAN V / Volta):** peak FP32 throughput and DRAM bandwidth are order-of-magnitude; this writeup uses **\~7450 GFLOP/s** and **\~652.8 GB/s**.

**Ridge (FLOP/byte) ≈ 7450 / 652.8 ≈ 11.4**.

### Theoretical arithmetic intensity (one full read/write of logical tensors)

| Tensor | Size | Bytes |
| :---- | :---- | ----: |
| Weights | 3×3×64×64 (floats) | 147,456 |
| Padded input | 226×226×64 (floats) | 13,075,456 |
| Output | 224×224×64 (floats) | 12,845,056 |
| **Total** |  | **26,067,968** ≈ 26.07 MiB |

**Theoretical arithmetic intensity:** **3,699,376,128 ÷ 26,067,968 ≈ 141.9** FLOP per byte 

### Measured DRAM traffic (NCU) and achieved AI

| Kernel | DRAM read+write (bytes, NCU) | FLOPs per byte (using same 3,699,376,128 FLOPs) |
| :---- | ----: | ----: |
| naive | 25.44×10⁶ | ≈ **145.4** |
| conv1\_optimized | 164.80×10⁶ | ≈ **22.5** |

---

## 5\. Optimizations

1. **Spatial tiling \+ shared input tile (8×8)**: one collaborative load of a **(10×10)×64** input tile per block, then many MACs from **shared** memory instead of re-reading global for every `ky,kx,ni` at every pixel.   
2. **Fixed 3×3 unrolling**: helps the compiler schedule loads and FMAs; moderate impact compared to tiling.  
3. **`float4` loads along `Ni` for input and weights**: reduces instruction count and can improve coalescing; meaningful on top of shared-memory reuse.
