#ifndef _SGEMM_IMPL_H_
#define _SGEMM_IMPL_H_

#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define FP_SIZE 16

// "fake" fp16 type that only has the correct data width.
using float16_t = uint16_t;

#if (FP_SIZE == 32)
using float_type = float;
#elif (FP_SIZE == 16)
using float_type = float16_t;
#endif

// Generate kernel for the Hopper-style SMEM-decoupled tensor core.  This uses
// asynchronous HGMMA and HGMMA_WAIT instructions.
#define TENSOR_HOPPER 0

// Constraints on parameters:
// * Memory:
//   (BM + BN) * BK * sizeof(T) <= sharedmem size.
//   BM * BK == BN * BK >= threadblock size >= NT * CORES_PER_CLUSTER
//     When larger, the kernel runs a sequential loop to read into sharedmem;
//     but smaller case is not handled.
// * Compute:
//   ( M* N) / (TM*TN) == grid size >= NC*NW*NT
//   (BM*BN) / (TM*TN) == threadblock size < NT * NW * CORES_PER_CLUSTER
//   (BM*BN) / (TM*TN) == threadblock size >= NT * CORES_PER_CLUSTER
// * Combining BM * BK >= (BM*BN) / (TM*TN) == threadblock yields
//   BM <= BK*TM*TN
#if (TENSOR_HOPPER == 1)
#define BM 128
#define BN 64
#if (FP_SIZE == 32)
#define BK 64
#elif (FP_SIZE == 16)
#define BK 128
#else
#error "unsupported FP_SIZE"
#endif

#define WM 16
#define WN 16
#define TCM 16
#define TCN 16
#if (FP_SIZE == 32)
#define TCK 16
#elif (FP_SIZE == 16)
#define TCK 32
#else
#error "unsupported FP_SIZE"
#endif

#else // !HOPPER

#define BM ((NUM_CORES == 8) ? 128 : 64)
#define BN 64
#if (FP_SIZE == 32)
#define BK 64
#elif (FP_SIZE == 16)
#define BK 128
#else
#error "unsupported FP_SIZE"
#endif

#define WM 16
#define WN 8
#define TCM 8
#define TCN 8
#if (FP_SIZE == 32)
#define TCK 8
#elif (FP_SIZE == 16)
#define TCK 16
#else
#error "unsupported FP_SIZE"
#endif
#endif

#define WMITER (WM / TCM)
#define WNITER (WN / TCN)
#define ELEM_PER_THREAD (WM * WN / NUM_THREADS)

static_assert(WMITER * WNITER * TCM * TCN * NUM_WARPS * CORES_PER_CLUSTER ==
                  BM * BN,
              "tile parameter condition not met (1 threadblock per cluster)");

// number of loop around the inner 0..TCK..BK loop to simulate perfect-DRAM
// scenario
#define BK_LOOP 1
// Whether to transpose smem A tile at GMEM->SMEM (produce), or SMEM->RF
// (consume).  This is because the tensor core expects the A tile to be stored
// in column-major order in SMEM, so a transpose is necessary if A was stored
// row-major in GMEM.
//
// For correctness, only one of either should be 1.  E.g., PRODUCE 1 CONSUME 0
// generates the NN kernel where both A and B are stored row-major in GMEM.
// To model the case where the A matrix is already stored column-major in GMEM,
// set both to 0.
#define TRANSPOSE_AT_PRODUCE 0
#define TRANSPOSE_AT_CONSUME 0

// if 1, wmma_store() will not respect the register <-> matrix fragment mapping
// scheme and instead do a fast coalesced GMEM writes for move out.  This
// doesn't necessarily mean breaking correctness; it means that the final
// result matrix will be stored in a swizzled form in the global memory.
#define WMMA_STORE_FAST 0

#define GEMMINI_DMA 1
#define GEMMINI_DMA_FAST 1
#if SMEM_SIZE == 0x4000
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff001000)
#define SMEM_ADDR_Q2 ((float * const) 0xff002000)
#define SMEM_ADDR_Q3 ((float * const) 0xff003000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x80
#define SPAD_ADDR_Q2 0x100
#define SPAD_ADDR_Q3 0x180
#define BOUND_INST 0x400040004ULL
#elif SMEM_SIZE >= 0x10000
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff004000)
#define SMEM_ADDR_Q2 ((float * const) 0xff008000)
#define SMEM_ADDR_Q3 ((float * const) 0xff00c000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x200
#define SPAD_ADDR_Q2 0x400
#define SPAD_ADDR_Q3 0x600
#define BOUND_INST 0x800080008ULL
#else
#error Unsupported smem size
#endif

// timing markers
#define MARK_BEG() asm volatile ("slti x0, x1, -1047")
#define MARK_END() asm volatile ("slti x0, x1, -499")

enum class MemLayout {
  MN_major,
  K_major,
  block_row_major, // Gemmini DMA
};

inline constexpr void map_operand_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // Figure 7(a) in paper
  // row  0~ 3: threadgroups 0 and 2
  // row  4~ 7: threadgroups 4 and 6
  // row  8~11: threadgroups 1 and 3
  // row 12~15: threadgroups 5 and 7
  row = tid % 4;
  row += (tg * 8) % 16;
  row += (tg / 4) * 4;

  // B (column major)
  // NOTE: Matrix B mapping in Figure 7(a) is incorrect; below is the
  // corrected mapping:
  // col  0~ 3: threadgroups 0 and 1
  // col  4~ 7: threadgroups 4 and 5
  // col  8~11: threadgroups 2 and 3
  // col 12~15: threadgroups 6 and 7
  col = tid % 4;
  col += ((tg % 4) / 2) * 8;
  col += (tg / 4) * 4;
}

inline constexpr void map_operand_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // row  0~ 3: threadgroup 0
  // row  4~ 7: threadgroup 1
  row = tid % 4;
  row += tg * 4;

  // B (column major)
  // col  0~ 3: threadgroup 0
  // col  4~ 7: threadgroup 1
  col = tid % 4;
  col += tg * 4;
}

inline constexpr void map_operand(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_operand_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_operand_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

inline constexpr void map_c_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  // Figure 7(b), left
  col = ((tg % 4) / 2) * 8;
  row = (tg * 8) % 16;
  row += (tg / 4) * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  col = 0;
  row = tg * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c_8lanes_coalesced(const int tid, int &row, int &col) {
  const int tg = tid / 2;

  row = 0;
  col = tid;
}

inline constexpr void map_c(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_c_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    if constexpr (TENSOR_HOPPER || WMMA_STORE_FAST) {
      map_c_8lanes_coalesced(tid, row, col);
    } else {
      map_c_8lanes(tid, row, col);
    }
  } else {
    // FIXME: not allowed
  }
}

#define RISCV_CUSTOM3   0x7B

inline void vx_wmma(const int dest_reg) {
  if (dest_reg == 0) {
    asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
  } else {
    asm volatile (".insn r %0, 0, 0, x1, x0, x0" :: "i"(RISCV_CUSTOM3));
  }
}

inline void vx_wgmma(const uint32_t addr_a, const uint32_t addr_b) {
  // .insn r opcode6, func3, func7, rd, rs1, rs2
  // https://www.rowleydownload.co.uk/arm/documentation/gnu/as/RISC_002dV_002dFormats.html#RISC_002dV_002dFormats
  asm volatile(".insn r %0, 0, 0, x0, %1, %2" ::"i"(RISCV_CUSTOM3), "r"(addr_a),
               "r"(addr_b));
}

inline void vx_wgmma_wait() {
  // .insn r opcode6, func3, func7, rd, rs1, rs2
  // func3 == 1 encodes wait
  asm volatile (".insn r %0, 1, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
}

// Remap logical row/col coordinate of a matrix element to a memory index that
// follows the 2-level block-row-major layout that Gemmini DMA uses
template <bool use_dma, uint32_t dim_col>
inline constexpr std::pair<uint32_t, uint32_t>
remap_to_gemmini_dma_layout(const uint32_t logical_row,
                            const uint32_t logical_col) {
  static_assert(!use_dma || DIM == 8,
                "GEMMINI_DMA layout remapping code only written for DIM == 8");

  if constexpr (use_dma) {
    constexpr int dim_blocks_in_row = (dim_col / DIM);
    const uint32_t row =
        (logical_row / dim_blocks_in_row) * DIM + (logical_col / DIM);
    const uint32_t col =
        (logical_row % dim_blocks_in_row) * DIM + (logical_col % DIM);
    return {row, col};
  } else {
    // pass-through
    return {logical_row, logical_col};
  }
}

// `local_k` is assumed to be multiple of TCK
template <typename T, MemLayout layout,
          uint32_t leading_dim // stride in sizeof(T) between consecutive
                               // "rows" in the memory.  What a row is
                               // corresponds to whatever `layout` specifies.
                               // E.g., if layout == MN_major, leading_dim
                               // becomes the stride between the 1st M-dim
                               // vector and the 2nd M-dim vector.
          >
inline volatile const uint8_t *
generate_smem_addr_a(volatile const T *smem_A, const int local_k,
                     const int warp_row, const int wm_iter,
                     const int thread_in_warp) {
  asm volatile ("generate_smem_addr_a_start_%=:" :: );

  const int tid = thread_in_warp;
  const int tg = tid / 4;

  // @perf: this is duplicately computed in wmma_load_a and wmma_load_b
  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  // In fp16 mode, bit-pack two fp16 elements into each fp32 element, and do
  // data movement at the fp32 granularity.  Assuming that the matrix is stored
  // row-major in GMEM, the packed fp16 pairs belong to the same row,
  // neighboring columns; therefore, it essentially becomes equivalent to
  // moving a fp32 matrix whose column dimensions (dim_k/BK/k) are compressed
  // by a factor of two.
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  const int local_k_adjusted = local_k / packed_factor;

  static_assert((layout != MemLayout::K_major) || (FP_SIZE == 32),
                "fp16 is not really tested for K-major A layout");

  if constexpr (layout == MemLayout::K_major ||
                layout == MemLayout::block_row_major) {
    constexpr int smem_A_cols = leading_dim;

    // f8-f15 stores a single row of A
    const uint32_t smem_logical_row = WM * warp_row + TCM * wm_iter + row;
    const uint32_t smem_logical_col =
        local_k_adjusted + 0; /* FIXME: fp16 adjust necessary? */
    // if using Gemmini DMA, remap logical row/col to Gemmini's 2-level
    // block-row-major layout
    const auto [smem_row, smem_col] =
        remap_to_gemmini_dma_layout<layout == MemLayout::block_row_major,
                                    smem_A_cols>(smem_logical_row,
                                                 smem_logical_col);

    return reinterpret_cast<const volatile uint8_t *>(
        &reinterpret_cast<const volatile float *>(
            smem_A)[smem_A_cols * smem_row + smem_col]);
  } else if constexpr (layout == MemLayout::MN_major) {
    constexpr int smem_AS_cols = leading_dim;

    return reinterpret_cast<const volatile uint8_t *>(
        &reinterpret_cast<const volatile float *>(
            smem_A)[((local_k_adjusted + 0) * smem_AS_cols) +
                    (WM * warp_row + TCM * wm_iter) + row]);
  } else {
    static_assert(layout ==
                      MemLayout::K_major /* fake cond that is always false */,
                  "unsupported memory layout");
  }

  asm volatile ("generate_smem_addr_a_finish_%=:" :: );
}

template <typename T, MemLayout layout, uint32_t leading_dim>
inline void wmma_load_a(volatile const T *smem_A, const int local_k,
                        const int warp_row, const int wm_iter,
                        const int thread_in_warp) {
  asm volatile ("wmma_load_a_start_%=:" :: );

  if constexpr (layout == MemLayout::K_major ||
                layout == MemLayout::block_row_major) {
    const volatile uint8_t *smem_addr =
        generate_smem_addr_a<T, layout, leading_dim>(smem_A, local_k, warp_row,
                                                     wm_iter, thread_in_warp);
    // step to the next column
    // @perf: bank conflicts; threads read from different rows
    // below is correct for GEMMINI_DMA; smem_col is always a multiple of 8,
    // and the next 7 elements in the row are guaranteed to be consecutive in
    // the memory
    asm volatile("flw  f0, %0(%1)" ::"i"(0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" ::"i"(1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" ::"i"(2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" ::"i"(3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" ::"i"(4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" ::"i"(5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" ::"i"(6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" ::"i"(7 * sizeof(float)), "r"(smem_addr));
  } else if constexpr (layout == MemLayout::MN_major) {
    constexpr int smem_AS_cols = leading_dim;

    const volatile uint8_t *smem_addr =
        generate_smem_addr_a<T, layout, leading_dim>(smem_A, local_k, warp_row,
                                                     wm_iter, thread_in_warp);
    // f8-f15 stores a single row of A
    // threads read from different columns; no bank conflicts
    // asm volatile("flw  f0, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f1, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f2, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f3, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f4, %0(%1)" :: "i"(smem_AS_cols * 4 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f5, %0(%1)" :: "i"(smem_AS_cols * 5 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f6, %0(%1)" :: "i"(smem_AS_cols * 6 * sizeof(float)), "r"(smem_addr));
    // asm volatile("flw  f7, %0(%1)" :: "i"(smem_AS_cols * 7 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f0, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
    smem_addr += smem_AS_cols * 4 * sizeof(float);
    asm volatile("flw  f4, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
  } else {
    static_assert(layout ==
                      MemLayout::K_major /* fake cond that is always false */,
                  "unsupported memory layout");
  }

  asm volatile ("wmma_load_a_finish_%=:" :: );
}

// Convenience wrapper for wmma_load_a if tile layout is packed, i.e.
// leading_dim == col.
template <typename T, MemLayout layout, uint32_t tile_dim_m,
          uint32_t tile_dim_n, uint32_t tile_dim_k>
inline void wmma_load_a(volatile const T *smem_A, const int local_k,
                        const int warp_row, const int wm_iter,
                        const int thread_in_warp) {
  // In fp16 mode, bit-pack two fp16 elements into each fp32 element, and do
  // data movement at the fp32 granularity.  Assuming that the matrix is stored
  // row-major in GMEM, the packed fp16 pairs belong to the same row,
  // neighboring columns; therefore, it essentially becomes equivalent to
  // moving a fp32 matrix whose column dimensions (dim_k/BK/k) are compressed
  // by a factor of two.
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  constexpr int tile_dim_k_adjusted = tile_dim_k / packed_factor;
  constexpr int leading_dim = (layout == MemLayout::K_major)
                                  ? tile_dim_k_adjusted
                                  : tile_dim_m;

  wmma_load_a<T, layout, leading_dim>(smem_A, local_k, warp_row, wm_iter,
                                      thread_in_warp);
}

// `local_k` is assumed to be multiple of TCK
template <typename T, MemLayout layout, uint32_t leading_dim,
          uint32_t tile_dim_k>
inline volatile const uint8_t *
generate_smem_addr_b(const volatile T *smem_B, const int local_k,
                     const int warp_col, const int wn_iter,
                     const int thread_in_warp) {
  asm volatile ("generate_smem_addr_b_start_%=:" :: );

  static_assert(
      layout == MemLayout::MN_major || layout == MemLayout::block_row_major,
      "only N-major or block-row-major layout are supported for the B tile");

  const int tid = thread_in_warp;
  const int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  // see comment in wmma_load_a
  constexpr int packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);
  constexpr int tile_dim_k_adjusted = tile_dim_k / packed_factor;
  const int local_k_adjusted = local_k / packed_factor;

  // B is stored N-major in smem
  constexpr int smem_B_cols = leading_dim;

  const uint32_t smem_logical_row = local_k_adjusted + 0;
  const uint32_t smem_logical_col = (WN * warp_col + TCN * wn_iter) + col;
  // if using Gemmini DMA, remap logical row/col to Gemmini's 2-level
  // block-row-major layout
  const auto [smem_row, smem_col] =
      remap_to_gemmini_dma_layout<layout == MemLayout::block_row_major,
                                  smem_B_cols>(smem_logical_row,
                                               smem_logical_col);

  return reinterpret_cast<const volatile uint8_t *>(
      &reinterpret_cast<const volatile float *>(
          smem_B)[smem_B_cols * smem_row + smem_col]);

  asm volatile ("generate_smem_addr_b_finish_%=:" :: );
}

template <typename T, MemLayout layout, uint32_t leading_dim,
          uint32_t tile_dim_k>
inline void wmma_load_b(const volatile T *smem_B, const int local_k,
                        const int warp_col, const int wn_iter,
                        const int thread_in_warp) {
  asm volatile ("wmma_load_b_start_%=:" :: );

  // B is stored N-major in smem
  constexpr int smem_B_cols = leading_dim;

  const volatile uint8_t *smem_addr =
      generate_smem_addr_b<T, layout, leading_dim, tile_dim_k>(
          smem_B, local_k, warp_col, wn_iter, thread_in_warp);

  // f8-f15 stores a single column of B
  // threads read from different columns; no bank conflicts
  if constexpr (layout == MemLayout::block_row_major) {
    // for the block-row-major layout, moving rows for the next 7 elements in
    // the same column is the same as moving DIM elements forward in the memory
    // because of the block-row-major layout
    asm volatile("flw  f8, %0(%1)" :: "i"(DIM * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f9, %0(%1)" :: "i"(DIM * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f10, %0(%1)" :: "i"(DIM * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f11, %0(%1)" :: "i"(DIM * 3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f12, %0(%1)" :: "i"(DIM * 4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f13, %0(%1)" :: "i"(DIM * 5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f14, %0(%1)" :: "i"(DIM * 6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f15, %0(%1)" :: "i"(DIM * 7 * sizeof(float)), "r"(smem_addr));
  } else {
    asm volatile("flw  f8, %0(%1)" :: "i"(smem_B_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f9, %0(%1)" :: "i"(smem_B_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f10, %0(%1)" :: "i"(smem_B_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f11, %0(%1)" :: "i"(smem_B_cols * 3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f12, %0(%1)" :: "i"(smem_B_cols * 4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f13, %0(%1)" :: "i"(smem_B_cols * 5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f14, %0(%1)" :: "i"(smem_B_cols * 6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw f15, %0(%1)" :: "i"(smem_B_cols * 7 * sizeof(float)), "r"(smem_addr));
  }

  asm volatile ("wmma_load_b_finish_%=:" :: );
}

template <typename T, MemLayout layout_a, MemLayout layout_b,
          uint32_t leading_dim_a, uint32_t leading_dim_b, uint32_t tile_dim_k>
inline void wgmma(volatile const T *smem_A, volatile const T *smem_B,
                  const int local_k, const int warp_row, const int warp_col) {
  asm volatile("wgmma_start_%=:" ::);

  const volatile uint8_t *addr_a =
      generate_smem_addr_a<T, layout_a, leading_dim_a>(smem_A, local_k,
                                                       warp_row, 0, 0);
  const volatile uint8_t *addr_b =
      generate_smem_addr_b<T, layout_b, leading_dim_b, tile_dim_k>(
          smem_B, local_k, warp_col, 0, 0);

  vx_wgmma(reinterpret_cast<uint32_t>(addr_a),
           reinterpret_cast<uint32_t>(addr_b));

  asm volatile("wgmma_finish_%=:" ::);
}

// Initialize the accumulator registers to zero before starting FMA operations
// with the tensor cores.
template <int accum_reg_set> inline void initialize_accum_regs() {
  if constexpr (accum_reg_set == 0) {
    asm volatile("fmv.w.x f16, x0");
    asm volatile("fmv.w.x f17, x0");
    asm volatile("fmv.w.x f18, x0");
    asm volatile("fmv.w.x f19, x0");
    asm volatile("fmv.w.x f20, x0");
    asm volatile("fmv.w.x f21, x0");
    asm volatile("fmv.w.x f22, x0");
    asm volatile("fmv.w.x f23, x0");
  } else {
    asm volatile("fmv.w.x f24, x0");
    asm volatile("fmv.w.x f25, x0");
    asm volatile("fmv.w.x f26, x0");
    asm volatile("fmv.w.x f27, x0");
    asm volatile("fmv.w.x f28, x0");
    asm volatile("fmv.w.x f29, x0");
    asm volatile("fmv.w.x f30, x0");
    asm volatile("fmv.w.x f31, x0");
  }
}

inline void initialize_all_regs() {
  asm volatile("fmv.w.x f0, x0");
  asm volatile("fmv.w.x f1, x0");
  asm volatile("fmv.w.x f2, x0");
  asm volatile("fmv.w.x f3, x0");
  asm volatile("fmv.w.x f4, x0");
  asm volatile("fmv.w.x f5, x0");
  asm volatile("fmv.w.x f6, x0");
  asm volatile("fmv.w.x f7, x0");
  asm volatile("fmv.w.x f8, x0");
  asm volatile("fmv.w.x f9, x0");
  asm volatile("fmv.w.x f10, x0");
  asm volatile("fmv.w.x f11, x0");
  asm volatile("fmv.w.x f12, x0");
  asm volatile("fmv.w.x f13, x0");
  asm volatile("fmv.w.x f14, x0");
  asm volatile("fmv.w.x f15, x0");
  asm volatile("fmv.w.x f16, x0");
  asm volatile("fmv.w.x f17, x0");
  asm volatile("fmv.w.x f18, x0");
  asm volatile("fmv.w.x f19, x0");
  asm volatile("fmv.w.x f20, x0");
  asm volatile("fmv.w.x f21, x0");
  asm volatile("fmv.w.x f22, x0");
  asm volatile("fmv.w.x f23, x0");
  asm volatile("fmv.w.x f24, x0");
  asm volatile("fmv.w.x f25, x0");
  asm volatile("fmv.w.x f26, x0");
  asm volatile("fmv.w.x f27, x0");
  asm volatile("fmv.w.x f28, x0");
  asm volatile("fmv.w.x f29, x0");
  asm volatile("fmv.w.x f30, x0");
  asm volatile("fmv.w.x f31, x0");
}

// `C` is expected to be in N-major layout.
__attribute__((always_inline)) inline void
wmma_load_accum(const int thread_in_warp, const int warp_col,
                const int warp_row, const int wn_iter, const int wm_iter,
                const int dim_n, const float *C) {
  asm volatile("wmma_load_accum_start_%=:" ::);

  const int tid = thread_in_warp;

  // these are [0, TCM/TCN)
  int tid_row = 0;
  int tid_col = 0;
  map_c(tid, tid_row, tid_col);

  int local_row = (WM * warp_row + TCM * wm_iter) + tid_row;
  int local_col = (WN * warp_col + TCN * wn_iter) + tid_col;

  // @copypaste from wmma_store
  // @perf: this likely causes a lot of gmem bank conflicts
  if (wm_iter == 0) {
    const uint8_t *addr = reinterpret_cast<const uint8_t *>(
        &C[dim_n * (local_row + 0) + (local_col + 0)]);
    const uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
    asm volatile("flw f16, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr));
    asm volatile("flw f17, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr));
    asm volatile("flw f18, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr_tworow));
    asm volatile("flw f19, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr_tworow));
    asm volatile("flw f20, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr));
    asm volatile("flw f21, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr));
    asm volatile("flw f22, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr_tworow));
    asm volatile("flw f23, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr_tworow));
  } else {
    const uint8_t *addr = reinterpret_cast<const uint8_t *>(
        &C[dim_n * (local_row + 0) + (local_col + 0)]);
    const uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
    asm volatile("flw f24, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr));
    asm volatile("flw f25, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr));
    asm volatile("flw f26, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr_tworow));
    asm volatile("flw f27, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr_tworow));
    asm volatile("flw f28, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr));
    asm volatile("flw f29, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr));
    asm volatile("flw f30, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr_tworow));
    asm volatile("flw f31, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr_tworow));
  }

  asm volatile("wmma_load_accum_finish_%=:" ::);
}

// Write out the matrix data stored in RF to memory
__attribute__((always_inline)) inline void
wmma_store(const int thread_in_warp, const int warp_col, const int warp_row,
           const int wn_iter, const int wm_iter, const int dim_n,
           float *write_addr) {
  asm volatile ("wmma_store_start_%=:" :: );

  const int tid = thread_in_warp;

  // these are [0, TCM/TCN)
  int tid_row = 0;
  int tid_col = 0;
  map_c(tid, tid_row, tid_col);

  int local_row = (WM * warp_row + TCM * wm_iter) + tid_row;
  int local_col = (WN * warp_col + TCN * wn_iter) + tid_col;

  // @perf: this likely causes a lot of gmem bank conflicts
  if (wm_iter == 0) {
    volatile uint8_t *addr = reinterpret_cast<volatile uint8_t *>(
        &write_addr[dim_n * (local_row + 0) + (local_col + 0)]);
    volatile uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
    if constexpr (!WMMA_STORE_FAST) {
      asm volatile("fsw f16, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr));
      asm volatile("fsw f17, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr));
      asm volatile("fsw f18, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr_tworow));
      asm volatile("fsw f19, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr_tworow));
      asm volatile("fsw f20, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr));
      asm volatile("fsw f21, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr));
      asm volatile("fsw f22, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr_tworow));
      asm volatile("fsw f23, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr_tworow));
    } else {
      asm volatile("fsw f16, %0(%1)" ::"i"(0 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f17, %0(%1)" ::"i"(1 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f18, %0(%1)" ::"i"(2 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f19, %0(%1)" ::"i"(3 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f20, %0(%1)" ::"i"(4 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f21, %0(%1)" ::"i"(5 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f22, %0(%1)" ::"i"(6 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f23, %0(%1)" ::"i"(7 * WN * sizeof(float)), "r"(addr));
    }
  } else {
    volatile uint8_t *addr = reinterpret_cast<volatile uint8_t *>(
        &write_addr[dim_n * (local_row + 0) + (local_col + 0)]);
    volatile uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
    if constexpr (!WMMA_STORE_FAST) {
      asm volatile("fsw f24, %0(%1)" ::"i"(0 * sizeof(float)), "r"(addr));
      asm volatile("fsw f25, %0(%1)" ::"i"(1 * sizeof(float)), "r"(addr));
      asm volatile("fsw f26, %0(%1)" ::"i"(2 * sizeof(float)), "r"(addr_tworow));
      asm volatile("fsw f27, %0(%1)" ::"i"(3 * sizeof(float)), "r"(addr_tworow));
      asm volatile("fsw f28, %0(%1)" ::"i"(4 * sizeof(float)), "r"(addr));
      asm volatile("fsw f29, %0(%1)" ::"i"(5 * sizeof(float)), "r"(addr));
      asm volatile("fsw f30, %0(%1)" ::"i"(6 * sizeof(float)), "r"(addr_tworow));
      asm volatile("fsw f31, %0(%1)" ::"i"(7 * sizeof(float)), "r"(addr_tworow));
    } else {
      asm volatile("fsw f24, %0(%1)" ::"i"(0 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f25, %0(%1)" ::"i"(1 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f26, %0(%1)" ::"i"(2 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f27, %0(%1)" ::"i"(3 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f28, %0(%1)" ::"i"(4 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f29, %0(%1)" ::"i"(5 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f30, %0(%1)" ::"i"(6 * WN * sizeof(float)), "r"(addr));
      asm volatile("fsw f31, %0(%1)" ::"i"(7 * WN * sizeof(float)), "r"(addr));
    }
  }

  asm volatile ("wmma_store_finish_%=:" :: );
}

// Write out the matrix data stored in RF to memory
__attribute__((always_inline)) inline void
wgmma_store(const int thread_in_warp, const int warp_col, const int warp_row,
            const int dim_n, float *write_addr) {
  asm volatile ("wgmma_store_start_%=:" :: );

  const int tid = thread_in_warp;

  // these are [0, TCM/TCN)
  int tid_row = 0;
  int tid_col = 0;
  map_c(tid, tid_row, tid_col);

  // FIXME: WM and WN might be swapped here
  int local_row = WM * warp_row + tid_row;
  int local_col = WN * warp_col + tid_col;

  // FIXME: this is storing in M-major format
  volatile uint8_t *addr = reinterpret_cast<volatile uint8_t *>(
      &write_addr[dim_n * (local_row + 0) + (local_col + 0)]);
  volatile uint8_t *addr_tworow = addr + (2 * dim_n) * sizeof(float);
  asm volatile("fsw f0, %0(%1)" ::"i"(0 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f1, %0(%1)" ::"i"(1 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f2, %0(%1)" ::"i"(2 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f3, %0(%1)" ::"i"(3 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f4, %0(%1)" ::"i"(4 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f5, %0(%1)" ::"i"(5 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f6, %0(%1)" ::"i"(6 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f7, %0(%1)" ::"i"(7 * WM * sizeof(float)), "r"(addr));

  asm volatile("fsw  f8, %0(%1)" ::"i"( 8 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw  f9, %0(%1)" ::"i"( 9 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f10, %0(%1)" ::"i"(10 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f11, %0(%1)" ::"i"(11 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f12, %0(%1)" ::"i"(12 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f13, %0(%1)" ::"i"(13 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f14, %0(%1)" ::"i"(14 * WM * sizeof(float)), "r"(addr));
  asm volatile("fsw f15, %0(%1)" ::"i"(15 * WM * sizeof(float)), "r"(addr));

  asm volatile("fsw f16, %0(%1)" ::"i"((0 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f17, %0(%1)" ::"i"((1 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f18, %0(%1)" ::"i"((2 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f19, %0(%1)" ::"i"((3 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f20, %0(%1)" ::"i"((4 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f21, %0(%1)" ::"i"((5 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f22, %0(%1)" ::"i"((6 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f23, %0(%1)" ::"i"((7 * WM + 8) * sizeof(float)), "r"(addr));

  asm volatile("fsw f24, %0(%1)" ::"i"(( 8 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f25, %0(%1)" ::"i"(( 9 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f26, %0(%1)" ::"i"((10 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f27, %0(%1)" ::"i"((11 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f28, %0(%1)" ::"i"((12 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f29, %0(%1)" ::"i"((13 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f30, %0(%1)" ::"i"((14 * WM + 8) * sizeof(float)), "r"(addr));
  asm volatile("fsw f31, %0(%1)" ::"i"((15 * WM + 8) * sizeof(float)), "r"(addr));

  asm volatile ("wgmma_store_finish_%=:" :: );
}

__attribute__((convergent)) inline void
threadblock_barrier(const uint32_t barrier_id, const uint32_t count) {
  asm volatile("" ::: "memory");
  vx_fence();
  vx_barrier(barrier_id, count);
}

// Move a single matrix tile from global memory (GMEM) to shared memory (SMEM).
// `dim_major`: major dimension of the matrix in GMEM, e.g. if K-major, K; or
// MN-major, M/N.
//
// Note that there's not a single way to specify a layout of the matrix.
// Identifying a matrix to be K-major and specifying the mn_index of a tile,
// is equivalent to identifying it as MN-major and specifying the k_index
// (provided `dim_major` is set accordingly).
template <typename T,
          MemLayout gmem_layout,           // memory layout of the GMEM tile
          MemLayout smem_layout,           // memory layout of the GMEM tile
          uint32_t tile_dim_mn,            // row dimension of the SMEM tile
          uint32_t tile_dim_k,             // column dimension of the SMEM tile
          uint32_t threads_per_threadblock // this needs to be
                                           // compile-time in order
                                           // to do inline assembly with
                                           // constant memory offsets
          >
__attribute__((always_inline)) inline void
load_tile_to_smem(const uint32_t dim_major, const uint32_t mn_index,
                  const uint32_t k_index, const T *global_addr,
                  volatile T *local_addr, const uint32_t tid_in_threadblock) {
  asm volatile("load_tile_to_smem_start_%=:" ::);

  // In fp16 mode, bit-pack two fp16 elements into each fp32 element, and do
  // data movement at the fp32 granularity.  The tensor core hardware assumes
  // the fp16 elements are contiguously stored along the K-dimension;
  // therefore, this essentially becomes equivalent to a fp32 GEMM where the
  // K-dimension is shrinked by the factor of two.
  constexpr uint32_t packed_factor = (std::is_same_v<T, float16_t> ? 2 : 1);

  constexpr uint32_t tile_dim_k_packed = tile_dim_k / packed_factor;
  constexpr uint32_t gmem_dim_row =
      (gmem_layout == MemLayout::K_major) ? tile_dim_mn : tile_dim_k_packed;
  constexpr uint32_t gmem_dim_col =
      (gmem_layout == MemLayout::K_major) ? tile_dim_k_packed : tile_dim_mn;
  constexpr uint32_t smem_dim_col =
      (smem_layout == MemLayout::K_major) ? tile_dim_k_packed : tile_dim_mn;

  const uint32_t dim_major_ =
      (gmem_layout == MemLayout::K_major) ? dim_major / packed_factor : dim_major;

  // threads in the threadblock always do contiguous accesses in the gmem
  const uint32_t local_row_gmem = tid_in_threadblock / gmem_dim_col;
  const uint32_t local_col_gmem = tid_in_threadblock % gmem_dim_col;

  constexpr bool transposed_write = (gmem_layout != smem_layout);
  // if transposed, threads write to smem in reversed col/row
  const uint32_t local_row_smem =
      transposed_write ? local_col_gmem : local_row_gmem;
  const uint32_t local_col_smem =
      transposed_write ? local_row_gmem : local_col_gmem;

  const uint32_t global_row_mn_major = tile_dim_k_packed * k_index + local_row_gmem;
  const uint32_t global_col_mn_major = gmem_dim_col * mn_index + local_col_gmem;
  const uint32_t global_row_k_major = gmem_dim_row * mn_index + local_row_gmem;
  const uint32_t global_col_k_major = tile_dim_k_packed * k_index + local_col_gmem;
  const uint32_t global_row = (gmem_layout == MemLayout::K_major)
                                  ? global_row_k_major
                                  : global_row_mn_major;
  const uint32_t global_col = (gmem_layout == MemLayout::K_major)
                                  ? global_col_k_major
                                  : global_col_mn_major;

  const float *global = reinterpret_cast<const float *>(global_addr) +
                        dim_major_ * global_row + global_col;
  volatile float *local = reinterpret_cast<volatile float *>(local_addr) +
                          smem_dim_col * local_row_smem + local_col_smem;

  constexpr uint32_t row_stride = threads_per_threadblock / gmem_dim_col;

  static_assert(row_stride * 8 <= gmem_dim_row,
                "manual loop unrolling condition not met; tile row dimension "
                "is too shallow");
  static_assert((gmem_dim_row % (row_stride * 8)) == 0,
                "manual loop unrolling condition not met; tile row dimension "
                "should be power-of-two");

#pragma GCC unroll 1
  // loop-unrolled flw/fsw to increase reuse distance and IPC
  for (uint32_t load_offset = 0; load_offset < gmem_dim_row;
       load_offset += row_stride * 8) {
    // equivalent code:
    //
    // *local = *global;
    // global += dim_major * row_stride;
    // local += BN * row_stride;

    // read same-column elements into fp registers
    asm volatile("flw ft0, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft1, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft2, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft3, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft4, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft5, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft6, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;
    asm volatile("flw ft7, (%0)" ::"r"(global));
    global += dim_major_ * row_stride;

    // need to branch because address offset constant in the inline assembly
    // cannot be larger than a certain limit
    if constexpr (!transposed_write) {
      // asm volatile("fsw ft0, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
      //                                      sizeof(float)),
      //              "r"(local));
      // asm volatile("fsw ft1, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
      //                                      sizeof(float)),
      //              "r"(local));
      // local += smem_dim_col * row_stride * 2;
      // asm volatile("fsw ft2, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
      //                                      sizeof(float)),
      //              "r"(local));
      // asm volatile("fsw ft3, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
      //                                      sizeof(float)),
      //              "r"(local));
      // local += smem_dim_col * row_stride * 2;
      // asm volatile("fsw ft4, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
      //                                      sizeof(float)),
      //              "r"(local));
      // asm volatile("fsw ft5, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
      //                                      sizeof(float)),
      //              "r"(local));
      // local += smem_dim_col * row_stride * 2;
      // asm volatile("fsw ft6, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
      //                                      sizeof(float)),
      //              "r"(local));
      // asm volatile("fsw ft7, %0(%1)" ::"i"(smem_dim_col * row_stride * 1 *
      //                                      sizeof(float)),
      //              "r"(local));
      // local += smem_dim_col * row_stride * 2;

      asm volatile("fsw ft0, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft1, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft2, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft3, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft4, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft5, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft6, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
      asm volatile("fsw ft7, %0(%1)" ::"i"(smem_dim_col * row_stride * 0 *
                                           sizeof(float)),
                   "r"(local));
      local += smem_dim_col * row_stride;
    } else {
      // currently, tensor core hardware only supports MN-major SMEM tile
      // layout for correct results
      static_assert(gmem_layout == MemLayout::K_major);
      static_assert(smem_layout == MemLayout::MN_major);

      asm volatile("fsw ft0, %0(%1)" ::"i"(row_stride * 0 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft1, %0(%1)" ::"i"(row_stride * 1 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft2, %0(%1)" ::"i"(row_stride * 2 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft3, %0(%1)" ::"i"(row_stride * 3 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft4, %0(%1)" ::"i"(row_stride * 4 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft5, %0(%1)" ::"i"(row_stride * 5 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft6, %0(%1)" ::"i"(row_stride * 6 * sizeof(float)),
                   "r"(local));
      asm volatile("fsw ft7, %0(%1)" ::"i"(row_stride * 7 * sizeof(float)),
                   "r"(local));
      local += row_stride * 8;
    }
  }

  asm volatile("load_tile_to_smem_finish_new_%=:" ::);
}

// Do a single tile*tile matrix multiplication using the matrix data stored in
// SMEM.  Useful in fused kernels where GEMMs are done at a per-tile scope.
template <typename T,
          MemLayout layout_a, // memory layout of `local_a`
          MemLayout layout_b, // memory layout of `local_b`
          uint32_t tile_dim_m, uint32_t tile_dim_n, uint32_t tile_dim_k,
          uint32_t leading_dim_a,    // if zero, assumes packed layout, i.e. row
                                     // stride == col.
          uint32_t leading_dim_b,    // if zero, assumes packed layout, i.e. row
                                     // stride == col.
          bool load_accum = false,   // if true, load the accumulation registers
                                     // with `local_c`.  used for the (C + A*B)
                                     // operation
          bool write_to_mem = false  // if true, write the single result tile to
                                     // the memory at a given address
          >
__attribute__((always_inline)) inline void thread_block_gemm_single_tile(
    const T *local_a, const T *local_b, const T *local_c, T *result_addr,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock,
    const uint32_t threadblocks_per_cluster,
    const uint32_t threadblock_id_in_cluster) {
  // no double-buffering
  // FIXME: duplicated from thread_block_gemm
  const uint32_t threads_per_warpgroup = threads_per_threadblock;
  const uint32_t warp_id_in_warpgroup = tid_in_threadblock / NUM_THREADS;
  const uint32_t warp_row = warp_id_in_warpgroup / (tile_dim_n / WN);
  const uint32_t warp_col = warp_id_in_warpgroup % (tile_dim_n / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

#if 1
  if constexpr (TENSOR_HOPPER) {
#pragma GCC unroll 1
    for (int i = 0; i < BK_LOOP; i++) {
#pragma GCC unroll 4
     for (uint32_t local_k = 0; local_k < tile_dim_k; local_k += TCK) {
       wgmma<T, layout_a, layout_b, leading_dim_a, tile_dim_n, tile_dim_k>(
           local_a, local_b, local_k, warp_row, warp_col);
      }
    }
  } else {
    // TODO: it would be useful if this bit is split out into a function, so
    // that preloading accumulation tile can be used for full GEMMs at the start
    // of the K-loop.
    if constexpr (load_accum) {
#pragma GCC unroll
      for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll
        for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
          wmma_load_accum(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter,
                          tile_dim_n, local_c);
        }
      }
    }

#pragma GCC unroll 1
    for (int i = 0; i < BK_LOOP; i++) {
#pragma GCC unroll 4
      for (uint32_t local_k = 0; local_k < tile_dim_k; local_k += TCK) {
#pragma GCC unroll 2
        for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
          // SMEM -> RF
          static_assert(leading_dim_b == 0,
                        "leading_dim for wmma_load_b is not implemented yet");
          wmma_load_b<T, layout_b, tile_dim_n,
                      tile_dim_k /*leading_dim_b is TODO */>(
              local_b, local_k, warp_col, wn_iter, tid_in_warp);
#pragma GCC unroll 2
          for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
            // SMEM -> RF
            if constexpr (leading_dim_a == 0) {
              wmma_load_a<T, layout_a, tile_dim_m, tile_dim_n, tile_dim_k>(
                  local_a, local_k, warp_row, wm_iter, tid_in_warp);
            } else {
              wmma_load_a<T, layout_a, leading_dim_a>(
                  local_a, local_k, warp_row, wm_iter, tid_in_warp);
            }
            // perform mma
            vx_wmma(wm_iter);
          }
        }
      }
    }
  }
#endif

  // if constexpr (GEMMINI_DMA) {
  //   // Call gemmini fence at the end of the loop to overlap dma & wmma.
  //   // Usually, by this time, dma has finished the copy so that this
  //   // becomes a no-op.
  //   if (tid_in_threadblock == 0) {
  //     gemmini_fence();
  //   }

  //   // reconverge after mmio
  //   threadblock_barrier(threadblock_id_in_cluster,
  //                       warps_per_threadblock_per_core);
  // }

  if constexpr (write_to_mem) {
    // need to protect smem reads in the earlier step from writes in below,
    // especially when the destination address overlaps with the source address
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

#pragma GCC unroll
    for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll
      for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
        wmma_store(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter, tile_dim_n,
                   result_addr);
      }
    }
  }
}

template <
    typename T, uint32_t threads_per_threadblock, bool write_to_gmem = true,
    // by default, A/B tiles are placed at the start of the smem
    uint32_t smem_a_offset = 0,      // byte offset of A tile in shared
                                     // memory
    uint32_t smem_a_dbuf_offset = 0, // byte offset of A
                                     // double-buffer tile in shared
                                     // memory
    uint32_t smem_b_offset = sizeof(T) * BM * BK, // byte offset of B tile
                                                      // in shared memory
    uint32_t smem_b_dbuf_offset = sizeof(T) * BM *
                                  BK // byte offset of B double-buffer
                                     // tile in shared memory
    >
inline void thread_block_gemm(const T *A, const T *B, float *C,
                              const uint32_t dim_m, const uint32_t dim_n,
                              const uint32_t dim_k,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threadblocks_per_cluster,
                              const uint32_t threadblock_id_in_cluster,
                              uint8_t *sharedmem_per_threadblock) {
  const uint32_t threads_per_warpgroup = threads_per_threadblock;
  const uint32_t warp_id_in_warpgroup = tid_in_threadblock / NUM_THREADS;
  const uint32_t warp_row = warp_id_in_warpgroup / (BN / WN);
  const uint32_t warp_col = warp_id_in_warpgroup % (BN / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  T *local_a = reinterpret_cast<T *>(sharedmem_per_threadblock + smem_a_offset);
  T *local_a_buf =
      reinterpret_cast<T *>(sharedmem_per_threadblock + smem_a_dbuf_offset);
  T *local_b = reinterpret_cast<T *>(sharedmem_per_threadblock + smem_b_offset);
  T *local_b_buf =
      reinterpret_cast<T *>(sharedmem_per_threadblock + smem_b_dbuf_offset);

  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);

#if (GEMMINI_DMA == 1)
  if (tid_in_threadblock == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose,
    // b_transpose);

    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 1);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);

    gemmini_fence();
  }
#endif

  // divide rows (M) by the number of threadblocks
  const uint32_t dim_m_range = (dim_m / threadblocks_per_cluster);
  const uint32_t dim_m_start = dim_m_range * threadblock_id_in_cluster;
  const uint32_t block_m_start = dim_m_start / BM;
  const uint32_t block_m_end = (dim_m_start + dim_m_range) / BM;

#pragma GCC unroll 1
  for (uint32_t block_m = block_m_start; block_m < block_m_end; block_m++) {
#pragma GCC unroll 1
    for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {
      asm volatile ("loop_mn_start_%=:" :: );

      if constexpr (TENSOR_HOPPER) {
        initialize_all_regs();
      } else {
        // clear out accumulators
        initialize_accum_regs<0>();
        initialize_accum_regs<1>();
      }

      if constexpr (GEMMINI_DMA) {
        // pipeline initiation
        if (block_m == 0 && block_n == 0) {
          if (tid_in_threadblock == 0) {
            // configure dma gmem address to load from
            ROCC_INSTRUCTION_RS1_RS2(
                XCUSTOM_ACC,
                (uint64_t)(A + block_m * BM * dim_k + /*block_k:*/ 0 * BK),
                (uint64_t)(B + /*block_k:*/ 0 * BK * dim_n + block_n * BN),
                k_LOOP_WS_CONFIG_ADDRS_AB)
            // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
            GEMMINI_CISC_CMD_R((dim_n << 20) | (dim_k << 8) | GEMMINI_CISC_SET_AB_STRIDE);
            gemmini_fence();

            GEMMINI_CISC_CMD_R((11 << 16) | (0 << 8) | GEMMINI_CISC_LOAD_TO_HEXADECILES);
            gemmini_fence();

#if 0
          // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
          // FIXME: block_k is 0 for two times
          sp_tiled_matmul_full_spad_ws(
#if 1
              SPAD_ADDR_Q0, SPAD_ADDR_Q1,
#else
              (/*block_k:*/ 0 & 1) ? SPAD_ADDR_Q2 : SPAD_ADDR_Q0,
              (/*block_k:*/ 0 & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q1,
#endif
              /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q3,
              /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
              /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
          gemmini_fence();
#endif
          }

          threadblock_barrier(threadblock_id_in_cluster,
                              warps_per_threadblock_per_core);
        }
      }

#pragma GCC unroll 1
      for (uint32_t block_k = 0; (block_k * BK) < dim_k; block_k++) {
        asm volatile("loop_k_start_%=:" ::);

        MARK_BEG();

        // producer code: GMEM->SMEM memory movement
        // ---------------------------------------------------------------------
        //
        // this is either done using DMA or SIMT cores depending on GEMMINI_DMA

#if (GEMMINI_DMA == 1)
        if (tid_in_threadblock == 0) {
          asm volatile("next_index_start_%=:" ::);

          const uint32_t next_block_k =
              ((block_k + 1) * BK == dim_k) ? 0 : block_k + 1;
          const uint32_t next_block_n =
              (next_block_k == 0)
                  ? (((block_n + 1) * BN == dim_n) ? 0 : block_n + 1)
                  : block_n;
          const uint32_t next_block_m =
              (next_block_n == 0)
                  ? (((block_m + 1) == block_m_end) ? block_m_start /*unused*/
                                                    : block_m + 1)
                  : block_m;

          asm volatile("next_index_end_%=:" ::);

          // configure dma gmem address to load from
          ROCC_INSTRUCTION_RS1_RS2(
              XCUSTOM_ACC,
              (uint64_t)(A + next_block_m * BM * dim_k + next_block_k * BK),
              (uint64_t)(B + next_block_k * BK * dim_n + next_block_n * BN),
              k_LOOP_WS_CONFIG_ADDRS_AB)
          // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
          GEMMINI_CISC_CMD_R((dim_n << 20) | (dim_k << 8) | 8);
          // gemmini_fence();

          // block_k is even: opcode 11 (write to local_a_buf)
          // block_k is odd:  opcode 10 (write to local_a)
          //
          // FIXME: This depends on (dim_k / BK) being an even number, since
          // the last iteration of the k-loop is prefetching for the first
          // iteration of the n-loop.  The ping-poing indexing has to match for
          // the two loop end to connect.
          const uint32_t a_hexadecile = 4 - ((block_k & 1) * 4);
          const uint32_t b_hexadecile = a_hexadecile + 11;
          GEMMINI_CISC_CMD_R((b_hexadecile << 16) | (a_hexadecile << 8) | GEMMINI_CISC_LOAD_TO_HEXADECILES);
          // // TODO: branch is probably slow
          // if (block_k & 1) {
          //   GEMMINI_CISC_CMD_I(12);
          // } else { // block_k == 0 is here
          //   GEMMINI_CISC_CMD_I(13);
          // }

          // configure loop iteration bounds
          // FIXME: shouldn't be necessary
          // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST,
          // k_LOOP_WS_CONFIG_BOUNDS) ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
          // SPAD_ADDR_Q0, SPAD_ADDR_Q1, k_LOOP_WS_CONFIG_SPAD_AB)
          // ROCC_INSTRUCTION_RS1_RS2(
          //     XCUSTOM_ACC,
          //     ((uint64_t)(/*a_spad_id:*/ 0) << 18) |
          //         ((uint64_t)(/*b_spad_id:*/ 0) << 16) |
          //         ((uint64_t)(/*act:0*/ 0) << 8) | ((/*low_D:*/ 0) << 2) |
          //         ((/*full_C:*/ 0) << 1) | (/*ex_accumulate:*/ 0),
          //     ((uint64_t)(/*C_spad_addr:*/ A) << 32) | 0x200U | (skips) |
          //         ((/*is_resadd*/ 0) << 2) | ((/*B_transpose:*/ 0) << 1) |
          //         (/*A_transpose:*/ 1),
          //     k_LOOP_WS)
          // gemmini_fence();

#if 0
          uint32_t spad_a_produce;
          uint32_t spad_b_produce;
          const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
          const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
          spad_a_produce =
              ((mask_odd & (SPAD_ADDR_Q0)) | (mask_even & (SPAD_ADDR_Q2)));
          spad_b_produce =
              ((mask_odd & (SPAD_ADDR_Q1)) | (mask_even & (SPAD_ADDR_Q3)));
          // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
          // FIXME: block_k is 0 for two times
          sp_tiled_matmul_full_spad_ws(
              spad_a_produce,
              spad_b_produce,
              /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
              /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
              /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
#endif
        }

        // reconverge after mmio divergence
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
#else
        // move A
        if constexpr (!TRANSPOSE_AT_PRODUCE) {
          load_tile_to_smem<T, MemLayout::MN_major, MemLayout::MN_major, BM, BK,
                            threads_per_threadblock>(
              dim_m, block_m, block_k, A, local_a, tid_in_threadblock);
        } else {
          load_tile_to_smem<T, MemLayout::K_major, MemLayout::MN_major, BM, BK,
                            threads_per_threadblock>(
              dim_k, block_m, block_k, A, local_a, tid_in_threadblock);
        }

        // move B
        load_tile_to_smem<T, MemLayout::MN_major, MemLayout::MN_major, BN, BK,
                          threads_per_threadblock>(dim_n, block_n, block_k, B,
                                                   local_b, tid_in_threadblock);

        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
#endif

        // consumer code: SMEM->RF and compute
        // ----------------------------------------------------------------------
        // @perf: this loop spills to stack a lot because of all the flws in
        asm volatile("dbuf_sel_start_%=:" ::);
        const T *local_a_consume;
        const T *local_b_consume;
        if constexpr (GEMMINI_DMA) {
          // local_a_consume = (k_index % 2) ? local_a_buf : local_a;
          // local_b_consume = (k_index % 2) ? local_b_buf : local_b;
          // FIXME: swap multiply with bitshifts
          // const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
          // const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
          // local_a_consume = reinterpret_cast<T *>(
          //     (mask_odd & reinterpret_cast<uintmax_t>(local_a_buf)) |
          //     (mask_even & reinterpret_cast<uintmax_t>(local_a)));
          // local_b_consume = reinterpret_cast<T *>(
          //     (mask_odd & reinterpret_cast<uintmax_t>(local_b_buf)) |
          //     (mask_even & reinterpret_cast<uintmax_t>(local_b)));
          local_a_consume = local_a + (block_k & 1) *
                                          (smem_a_dbuf_offset - smem_a_offset) /
                                          sizeof(T);
          local_b_consume = local_b + (block_k & 1) *
                                          (smem_b_dbuf_offset - smem_b_offset) /
                                          sizeof(T);
        } else {
          // no double-buffering without DMA
          local_a_consume = local_a;
          local_b_consume = local_b;
        }

        asm volatile("dbuf_sel_end_%=:" ::);

        constexpr MemLayout layout_a =
            GEMMINI_DMA ? (GEMMINI_DMA_FAST ? MemLayout::MN_major
                                            : MemLayout::block_row_major)
                        : (TRANSPOSE_AT_CONSUME ? MemLayout::K_major
                                                : MemLayout::MN_major);
        constexpr MemLayout layout_b =
            GEMMINI_DMA ? (GEMMINI_DMA_FAST ? MemLayout::MN_major
                                            : MemLayout::block_row_major)
                        : MemLayout::MN_major;
        thread_block_gemm_single_tile<T, layout_a, layout_b,
                                      BM, BN, BK, 0, 0,
                                      /*load_accum=*/false,
                                      /*write_to_mem=*/false>(
            local_a_consume, local_b_consume,
            static_cast<T *>(nullptr) /*ignore accum*/,
            static_cast<T *>(nullptr) /*ignore result*/, tid_in_threadblock,
            threads_per_threadblock, threadblocks_per_cluster,
            threadblock_id_in_cluster);

        if constexpr (GEMMINI_DMA) {
          // Call gemmini fence at the end of the loop to overlap dma & wmma.
          // Usually, by this time, dma has finished the copy so that this
          // becomes a no-op.
          if (tid_in_threadblock == 0) {
            gemmini_fence();
          }
        }

        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);

        MARK_END();

        asm volatile("loop_k_end_%=:" ::);
      }

      if constexpr (write_to_gmem) {
        asm volatile("move_out_start_%=:" ::);

        if constexpr (TENSOR_HOPPER) {
          // wait until all results are accumulated into the RF
          vx_wgmma_wait();

          float *global_offset_C = C + (BM * block_m) * dim_n + BN * block_n;
          wgmma_store(tid_in_warp, warp_col, warp_row, dim_n, global_offset_C);
        } else {
#pragma GCC unroll
          for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll
            for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
              float *global_offset_C =
                  C + (BM * block_m) * dim_n + BN * block_n;
              wmma_store(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter,
                         dim_n, global_offset_C);
            }
          }
        }

        asm volatile("move_out_end_%=:" ::);
      }

      asm volatile("loop_mn_end_%=:" ::);
    }
  }
}

#endif
