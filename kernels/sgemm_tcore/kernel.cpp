#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define MARK_BEG() asm volatile ("slti x0, x1, -1047")
#define MARK_END() asm volatile ("slti x0, x1, -499")

constexpr bool DEBUG = false;

// FIXME: doesn't take FLOAT_SIZE into account
template <uint32_t tile_dim_row, uint32_t tile_dim_col>
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

#pragma GCC unroll 1
  for (int row_offset = 0; row_offset < tile_dim_row;
       row_offset += warps_in_threadblock) {
    const uint32_t row = row_offset + warp_id;
    const uint32_t first_thread_offset = tile_dim_col * row;

    constexpr uint32_t per_row_iter = tile_dim_col / NUM_THREADS;
    uint32_t thread_offset = first_thread_offset + tid_in_warp;
#pragma GCC unroll
    for (int i = 0; i < per_row_iter; i++) {
      dest[thread_offset] = src[thread_offset];
      thread_offset += NUM_THREADS;
    }

    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);
  }

  asm volatile("threadblock_copy_tile_finish_%=:" ::);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

#ifdef RADIANCE
  constexpr uint32_t cores_per_cluster = CORES_PER_CLUSTER;
#else
  constexpr uint32_t cores_per_cluster = 1;
#endif

  constexpr uint32_t threads_per_threadblock_theoretical =
      (BM * BN) / (ELEM_PER_THREAD);
  constexpr uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * NUM_THREADS * NUM_WARPS;
  // cap maximum threadblock size to # of HW threads in cluster, to prevent
  // multiple "wave" invocations which slows down the kernel
  constexpr uint32_t threads_per_threadblock =
      (threads_per_threadblock_theoretical > hw_threads_per_cluster)
          ? hw_threads_per_cluster
          : threads_per_threadblock_theoretical;
  constexpr uint32_t threadblocks_per_cluster =
      hw_threads_per_cluster / threads_per_threadblock;
  constexpr uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_cluster =
      threadblock_id % threadblocks_per_cluster;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;
  const uint32_t problem_size = (dim_m * dim_n) / (ELEM_PER_THREAD);
  const uint32_t num_threadblocks = problem_size / threads_per_threadblock;

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  uint8_t *sharedmem_per_threadblock = reinterpret_cast<uint8_t *>(
      DEV_SMEM_START_ADDR +
      sizeof(float_type) * 2 * (2 * BM * BK) * threadblock_id_in_cluster);

  // NOTE: hardcoded
  constexpr uint32_t quartile = (128 << 10) >> 2; // 128KB / 4
  static_assert((quartile * 4) == SMEM_SIZE, "wrong quartile constant");

  MARK_BEG();

  constexpr uint32_t smem_a_offset = 0;
  constexpr uint32_t smem_a_dbuf_offset = 1 * quartile;
  constexpr uint32_t smem_b_offset =
      3 * quartile - BN * BK * sizeof(float_type);
  constexpr uint32_t smem_b_dbuf_offset =
      4 * quartile - BN * BK * sizeof(float_type);
  thread_block_gemm<float_type, threads_per_threadblock,
                    /*write_to_gmem=*/true,
                    /*smem_a_offset=*/smem_a_offset,
#ifdef GEMMINI_DMA
                    /*smem_a_dbuf_offset=*/smem_a_dbuf_offset,
                    /*smem_b_offset=*/smem_b_offset,
                    /*smem_b_dbuf_offset=*/smem_b_dbuf_offset
#else
                    /*smem_a_dbuf_offset=*/1 * BM * BK * sizeof(float_type),
                    /*smem_b_offset=*/2 * BM * BK * sizeof(float_type),
                    /*smem_b_dbuf_offset=*/(2 * BM * BK + BK * BN) * sizeof(float_type)
#endif
                    >((const float_type *)arg->addr_a,
                      (const float_type *)arg->addr_b, (float *)arg->addr_c,
                      arg->dim_m, arg->dim_n, arg->dim_k, tid_in_threadblock,
                      threadblocks_per_cluster, threadblock_id_in_cluster,
                      sharedmem_per_threadblock);

  MARK_END();

  float *gmem_tmp_d0 = reinterpret_cast<float *>(0xd0000000UL);
  float *gmem_tmp_d1 = reinterpret_cast<float *>(0xd1000000UL);
  float *gmem_tmp_d2 = reinterpret_cast<float *>(0xd2000000UL);
  float *gmem_tmp_d3 = reinterpret_cast<float *>(0xd3000000UL);

  const float *smem_A0 =
      reinterpret_cast<float *>(sharedmem_per_threadblock + smem_a_offset);
  const float *smem_A1 =
      reinterpret_cast<float *>(sharedmem_per_threadblock + smem_a_dbuf_offset);
  const float *smem_B0 =
      reinterpret_cast<float *>(sharedmem_per_threadblock + smem_b_offset);
  const float *smem_B1 =
      reinterpret_cast<float *>(sharedmem_per_threadblock + smem_b_dbuf_offset);
  // const float *smem_B = reinterpret_cast<float *>(
  //     sharedmem_per_threadblock + 2 * BM * BK * sizeof(float_type));

  if constexpr (DEBUG) {
    threadblock_barrier(threadblock_id_in_cluster,
                        warps_per_threadblock_per_core);

    thread_block_copy_tile<BM, BK>(smem_A0, gmem_tmp_d0, tid_in_threadblock,
                                   threads_per_threadblock,
                                   threadblock_id_in_cluster);
    thread_block_copy_tile<BM, BK>(smem_A1, gmem_tmp_d1, tid_in_threadblock,
                                   threads_per_threadblock,
                                   threadblock_id_in_cluster);
    thread_block_copy_tile<BK, BN>(smem_B0, gmem_tmp_d2, tid_in_threadblock,
                                   threads_per_threadblock,
                                   threadblock_id_in_cluster);
    thread_block_copy_tile<BK, BN>(smem_B1, gmem_tmp_d3, tid_in_threadblock,
                                   threads_per_threadblock,
                                   threadblock_id_in_cluster);
  }
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t problem_size = (arg->dim_m * arg->dim_n) / (ELEM_PER_THREAD);
  const uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps();
  // prevent launching more threads than the necessary problem size
  // TODO: this does not take into account multiple clusters
  const uint32_t grid_size = (problem_size > hw_threads_per_cluster)
                                 ? hw_threads_per_cluster
                                 : problem_size;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
