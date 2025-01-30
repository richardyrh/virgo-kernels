#ifndef _FLASH_IMPL_H_
#define _FLASH_IMPL_H_

#include <vx_spawn.h>
#include <float.h>

#define MARK_BEG() asm volatile ("slti x0, x1, -1047")
#define MARK_END() asm volatile ("slti x0, x1, -499")

#define B_ROW 64
#define B_COL 64
#define HEADDIM 64

#define ROW_REMAINDER_LOGIC

constexpr uint32_t ROWMAX_SETS = 3;
// constexpr bool WARP_SPECIALIZED = true;
// constexpr bool GEMMINI_WARP_SPECIALIZED = false;
// constexpr bool TENSOR_CORE = true;
constexpr bool WARP_SPECIALIZED = true;
constexpr bool GEMMINI_WARP_SPECIALIZED = false;
constexpr bool TENSOR_CORE = true;

// temporary safety stop for wrong configs
static_assert(NUM_CORES == 4);
static_assert(NUM_THREADS == 8);
static_assert(NUM_WARPS == 8);

inline void thread_block_init_sharedmem(const uint32_t tid_in_threadblock,
                                        const uint32_t threads_per_threadblock,
                                        float *smem_O, float *smem_rowmax,
                                        float *smem_rowsum,
                                        float *smem_O_row_scale) {
  asm volatile("threadblock_init_sharedmem_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;

  static_assert((B_ROW % NUM_THREADS) == 0,
                "B_ROW must be a multiple of NUM_THREADS");
  static_assert(B_ROW < (NUM_THREADS * CORES_PER_CLUSTER *
                         (NUM_WARPS / (WARP_SPECIALIZED ? 2 : 1))),
                "not enough warps to initialize rowmax/rowsum");

  // each thread initializes one element in rowmax/rowsum
  // multiple warps participate for the whole vector
  constexpr uint32_t needed_warps = B_ROW / NUM_THREADS;
  if (warp_id < needed_warps /* more warps in HW than needed? */) {
    uint32_t offset = NUM_THREADS * warp_id + tid_in_warp;
#pragma GCC unroll
    for (int i = 0; i < ROWMAX_SETS; i++) {
      smem_rowmax[offset + i * ROWMAX_SETS] = FLT_MIN;
    }
    smem_rowsum[offset] = 0.0f;
    smem_O_row_scale[offset] = 0.0f;
  }

  // each warp clears out a row of smem_O
  // FIXME: dedup this pattern
#pragma GCC unroll 1
  for (int row_offset = 0; row_offset < B_COL;
       row_offset += warps_in_threadblock) {
    const uint32_t row = row_offset + warp_id;
#ifdef ROW_REMAINDER_LOGIC
    if (row >= B_ROW) {
      // WARNING: the number of barrier calls have to exactly match that in the
      // outside of the branch to prevent stalls!! FIXME better proof this.
      continue;
    }
#endif

    uint32_t thread_offset = HEADDIM * row + tid_in_warp;
    constexpr uint32_t per_row_iter = HEADDIM / NUM_THREADS;
    const float one = 0.0f;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      smem_O[thread_offset] = 0.0f;
      thread_offset += NUM_THREADS;
    }
  }

  asm volatile("threadblock_init_sharedmem_finish_%=:" ::);
}

inline void thread_block_copy_rowmax(const float *src, float *dest,
                                     const uint32_t tid_in_threadblock,
                                     const uint32_t threads_per_threadblock,
                                     const uint32_t threadblock_id_in_cluster) {
  asm volatile("threadblock_copy_rowmax_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      warps_in_threadblock / CORES_PER_CLUSTER;

  // each thread copies one element in rowmax
  // multiple warps participate for the whole vector
  constexpr uint32_t num_warps = B_ROW / NUM_THREADS;
  if (warp_id < num_warps) {
    uint32_t offset = NUM_THREADS * warp_id + tid_in_warp;
    dest[offset] = src[offset];
  }

  if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
    threadblock_barrier(1, 7);
  } else {
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);
  }

  asm volatile("threadblock_copy_rowmax_finish_%=:" ::);
}

template <uint32_t dim_row, uint32_t dim_col, bool block_row_major = false>
inline void thread_block_copy_tile(const float *src, float *dest,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock,
                                   const uint32_t threadblock_id_in_cluster) {
  asm volatile("threadblock_copy_tile_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      warps_in_threadblock / CORES_PER_CLUSTER;

  // FIXME: dedup this pattern
#pragma GCC unroll 1
  for (int row_offset = 0; row_offset < dim_row;
       row_offset += warps_in_threadblock) {
    const uint32_t row = row_offset + warp_id;
#ifdef ROW_REMAINDER_LOGIC
    if (row >= B_ROW) {
      // WARNING: the number of barrier calls have to exactly match that in the
      // outside of the branch to prevent stalls!! FIXME better proof this.
      if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
        threadblock_barrier(1, 7);
      } else {
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
      }
      continue;
    }
#endif

    constexpr uint32_t per_row_iter = dim_col / NUM_THREADS;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const uint32_t col_offset = NUM_THREADS * i;
      const uint32_t col = col_offset + tid_in_warp;
      const auto [smem_row, smem_col] =
          remap_to_gemmini_dma_layout<block_row_major, B_COL>(row, col);
      const uint32_t smem_offset = B_COL * smem_row + smem_col;
      const uint32_t gmem_offset = B_COL * row + col;

      dest[gmem_offset] = src[smem_offset];
    }

    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }
  }

  asm volatile("threadblock_copy_tile_finish_%=:" ::);
}

template <int order>
inline float exponential_taylor_term(const float x) {
  asm volatile("exponential_taylor_term_start_%=:" ::);

  float res = 1.0f;

  if constexpr (order == 1) {
    res = x;
  } else if constexpr (order == 2) {
    res = x * x;
    res /= 2.0f;
  } else if constexpr (order == 3) {
    res = x * x * x;
    res /= 6.0f;
  }

  asm volatile("exponential_taylor_term_end_%=:" ::);
  return res;
}

template <bool block_row_major = false>
__attribute__((always_inline)) inline void thread_block_online_softmax(
    const float *smem_S, float *smem_P, const uint32_t tid_in_threadblock,
    const uint32_t threads_per_threadblock,
    const uint32_t threadblock_id_in_cluster, float *smem_scratchpad,
    float *smem_rowmax, float *smem_rowsum, float *smem_O_row_scale) {
  asm volatile("thread_block_online_softmax_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      warps_in_threadblock / CORES_PER_CLUSTER;

  float *smem_rowmax_this = smem_rowmax + B_ROW;

#pragma GCC unroll 1
  for (int row_offset = 0; row_offset < B_ROW;
       row_offset += warps_in_threadblock) {
    const uint32_t row = row_offset + warp_id;
#ifdef ROW_REMAINDER_LOGIC
    // if the number of warps doesn't exactly divide the number of rows,
    // early-exit to prevent out-of-bounds access
    if (row >= B_ROW) {
      // WARNING: the number of barrier calls have to exactly match that in the
      // outside of the branch to prevent stalls!! FIXME better proof this.
      if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
        threadblock_barrier(1, 7);
        threadblock_barrier(1, 7);
        threadblock_barrier(1, 7);
        threadblock_barrier(1, 7);
        threadblock_barrier(1, 7);
        threadblock_barrier(1, 7);
      } else {
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
        threadblock_barrier(threadblock_id_in_cluster,
                            warps_per_threadblock_per_core);
      }

      continue;
    }
#endif
    const uint32_t first_thread_offset = B_COL * row;

    // rowmax
    //
    // two-level tree reduction: reduce each row into NUM_THREADS intermediate
    // maxes, then reduce it down to one row max
    // one warp handles one row in tile

    constexpr uint32_t per_row_iter = B_COL / NUM_THREADS;
    // FIXME: threadblock_id needs to be in here too
    float *warp_smem = smem_scratchpad + (warp_id * NUM_THREADS);

// #define DUMB_ROWMAX
#ifdef DUMB_ROWMAX
    // FIXME remove
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    // no tree reduction; a single thread in a warp does serialized max across
    // the entire row
    if (tid_in_warp == 0) {
      float rowmax = smem_S[first_thread_offset];
#pragma GCC unroll 16
      for (int i = 0; i < B_COL; i++) {
        asm volatile("fmax.s %0, %1, %2"
                     : "=f"(rowmax)
                     : "f"(rowmax), "f"(smem_S[first_thread_offset + i]));
      }
      smem_rowmax_this[row] = rowmax;

      // update previous rowmax
      // i.e. mi_new = max(mi, mij)
      float prev_rowmax = smem_rowmax[row];
      // stage prev rowmax in scratchpad for warp-wide broadcast
      warp_smem[0] = prev_rowmax;
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(rowmax)
                   : "f"(rowmax), "f"(prev_rowmax));
      smem_rowmax[row] = rowmax;
    }

#else
    static_assert((B_COL % NUM_THREADS) == 0,
                  "B_COL must be a multiple of NUM_THREADS");
    float per_thread_max = FLT_MIN;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const uint32_t col_offset = NUM_THREADS * i;
      const uint32_t col = col_offset + tid_in_warp;
      const auto [smem_row, smem_col] =
          remap_to_gemmini_dma_layout<block_row_major, B_COL>(row, col);
      const uint32_t offset = B_COL * smem_row + smem_col;

      const float next = smem_S[offset];
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(per_thread_max)
                   : "f"(per_thread_max), "f"(next));
    }
    // stage per-thread max value in smem
    warp_smem[tid_in_warp] = per_thread_max;

    // sync writes to warp_smem
    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }

// #define PARALLEL_ROWMAX
#ifndef PARALLEL_ROWMAX
    // elect 0-th thread to reduce all other thread's values in the warp
    if (tid_in_warp == 0) {
      float rowmax = per_thread_max;
      for (int i = 1; i < NUM_THREADS; i++) {
        float other = warp_smem[i];
        asm volatile("fmax.s %0, %1, %2"
                     : "=f"(rowmax)
                     : "f"(rowmax), "f"(other));
      }
      smem_rowmax_this[row] = rowmax;

      // update previous rowmax
      // i.e. mi_new = max(mi, mij)
      float prev_rowmax = smem_rowmax[row];
      // stage prev rowmax in scratchpad for warp-wide broadcast
      warp_smem[0] = prev_rowmax;
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(rowmax)
                   : "f"(rowmax), "f"(prev_rowmax));
      smem_rowmax[row] = rowmax;
    }
#else
    if (warp_id < warps_in_threadblock / NUM_THREADS) {
      const uint32_t row = row_offset + NUM_THREADS * warp_id + tid_in_warp;
      float *const thread_smem = smem_scratchpad + (tid_in_warp * NUM_THREADS);
      float rowmax = FLT_MIN;
#pragma GCC unroll
      for (int i = 0; i < NUM_THREADS; i++) {
        const float f = thread_smem[i];
        asm volatile("fmax.s %0, %1, %2" : "=f"(rowmax) : "f"(rowmax), "f"(f));
      }
      smem_rowmax_this[row] = rowmax;

      // update previous rowmax
      // i.e. mi_new = max(mi, mij)
      float prev_rowmax = smem_rowmax[row];
      // stage prev rowmax in scratchpad for warp-wide broadcast
      thread_smem[0] = prev_rowmax;
      asm volatile("fmax.s %0, %1, %2"
                   : "=f"(rowmax)
                   : "f"(rowmax), "f"(prev_rowmax));
      smem_rowmax[row] = rowmax;
    }
#endif // PARALLEL_ROWMAX
#endif // DUMB_ROWMAX

    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }


    // broadcast prev rowmax to all threads in the warp
    // NOTE: memory consistency is a little sketchy here
    const float rowmax_prev = warp_smem[0];
    const float rowmax_this = smem_rowmax_this[row];

    // exponential
    //
    // B_ROW / (B_ROW * B_COL / (exp_elem * threads_per_threadblock))
    // const uint32_t row_stride =
    //     (exp_elem_per_thread * threads_per_threadblock) / B_COL;

    // broadcast updated rowmax to all threads in the warp
    const float rowmax_new = smem_rowmax[row];

    asm volatile("flashattn_exp_p_start_%=:" ::);

#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const uint32_t col_offset = NUM_THREADS * i;
      const uint32_t col = col_offset + tid_in_warp;
      const auto [smem_row, smem_col] =
          remap_to_gemmini_dma_layout<block_row_major, B_COL>(row, col);
      const uint32_t offset = B_COL * smem_row + smem_col;

      float f0 = smem_S[offset];

      f0 -= rowmax_new;

      // 2nd-order Taylor approximation
      float exp = 1.0f;
      exp += exponential_taylor_term<1>(f0);
      exp += exponential_taylor_term<2>(f0);

      // Store S transposed to the shared memory

      smem_P[offset] = exp;
    }

    asm volatile("flashattn_exp_p_end_%=:" ::);

    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }

    // rowsum
    //
    // two-level tree reduction, similar to rowmax

    asm volatile("flashattn_rowsum_start_%=:" ::);

    float per_thread_sum = 0.0f;

#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const uint32_t col_offset = NUM_THREADS * i;
      const uint32_t col = col_offset + tid_in_warp;
      const auto [smem_row, smem_col] =
          remap_to_gemmini_dma_layout<block_row_major, B_COL>(row, col);
      const uint32_t offset = B_COL * smem_row + smem_col;

      per_thread_sum += smem_P[offset];
    }
    // stage per-thread sum value in smem
    // FIXME: threadblock_id needs to be in here too
    warp_smem = smem_scratchpad + (warp_id * NUM_THREADS);
    warp_smem[tid_in_warp] = per_thread_sum;

    // sync writes to warp_smem
    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }

    // 0-th thread collects all other thread's values in the warp
    if (tid_in_warp == 0) {
      float rowsum = per_thread_sum;
      for (int iter = 1; iter < NUM_THREADS; iter++) {
        float other = warp_smem[iter];
        rowsum += other;
      }

      const float mi_prev = rowmax_prev;
      const float mi_this = rowmax_this;

      const float x = mi_prev - mi_this;
      // 2nd-order Taylor approximation
      float exp = 1.0f;
      exp += exponential_taylor_term<1>(x);
      exp += exponential_taylor_term<2>(x);

      // update rowsum
      const float rowsum_prev = smem_rowsum[row];
      float rowsum_new = exp * rowsum_prev + rowsum;

      smem_rowsum[row] = rowsum_new;
    }

    asm volatile("flashattn_rowsum_end_%=:" ::);

    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }

    // compute Oi rescale factor
    // FIXME: parallelize this across threads
    //
    asm volatile("flashattn_rescale_factor_start_%=:" ::);

#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const float mi_prev = rowmax_prev;
      const float mi_new = rowmax_new;

      const float x = mi_prev - mi_new;
      // 2nd-order Taylor approximation
      float exp = 1.0f;
      exp += exponential_taylor_term<1>(x);
      exp += exponential_taylor_term<2>(x);

      // @perf: div vs. expansion on e(-x)?
      smem_O_row_scale[row] = 1.0f / exp;
    }

    asm volatile("flashattn_rescale_factor_end_%=:" ::);

    if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
      threadblock_barrier(1, 7);
    } else {
      threadblock_barrier(threadblock_id_in_cluster,
                          warps_per_threadblock_per_core);
    }
  }

  asm volatile("thread_block_online_softmax_finish_%=:" ::);
}

template <bool block_row_major = false>
__attribute__((always_inline)) inline void thread_block_O_rescale(
    const float *smem_O_in, float *smem_O_out, const float *smem_O_row_scale,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock,
    const uint32_t threadblock_id_in_cluster) {
  asm volatile("thread_block_O_rescale_start_%=:" ::);

  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  const uint32_t warps_in_threadblock = threads_per_threadblock / NUM_THREADS;
  const uint32_t warps_per_threadblock_per_core =
      warps_in_threadblock / CORES_PER_CLUSTER;

#pragma GCC unroll 1
  for (int row_offset = 0; row_offset < B_ROW;
       row_offset += warps_in_threadblock) {
    const uint32_t row = row_offset + warp_id;
#ifdef ROW_REMAINDER_LOGIC
    if (row >= B_ROW) {
      // WARNING: the number of barrier calls have to exactly match that in the
      // outside of the branch to prevent stalls!! FIXME better proof this.
      continue;
    }
#endif

    constexpr uint32_t per_row_iter = HEADDIM / NUM_THREADS;

    // Oi rescale
    //
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      const uint32_t col_offset = NUM_THREADS * i;
      const uint32_t col = col_offset + tid_in_warp;
      const auto [smem_row, smem_col] =
          remap_to_gemmini_dma_layout<block_row_major, HEADDIM>(row, col);

      const uint32_t offset = HEADDIM * smem_row + smem_col;
      const float o = smem_O_in[offset];
      const float scale = smem_O_row_scale[row];
      smem_O_out[offset] = (o * scale);
    }
  }

  // reconverge after warp divergence
  if constexpr (!TENSOR_CORE && GEMMINI_WARP_SPECIALIZED) {
    threadblock_barrier(1, 7);
  } else {
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);
  }

  asm volatile("thread_block_O_rescale_finish_%=:" ::);
}

#endif
