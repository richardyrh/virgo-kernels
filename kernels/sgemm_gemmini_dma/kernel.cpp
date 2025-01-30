#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

// fp16 16x16
#define TILE_M 128
#define TILE_N 64
#define TILE_K 128

// ampere
// #define NUM_THREADS_IN_CLUSTER 512
// hopper
#define NUM_THREADS_IN_CLUSTER 256

// fp32 8x8
// #define TILE_M 64
// #define TILE_N 64
// #define TILE_K 64
// #define SMEM_ADDR_Q0 ((float * const) 0xff000000)
// #define SMEM_ADDR_Q1 ((float * const) 0xff004000)
// #define SMEM_ADDR_Q2 ((float * const) 0xff008000)
// #define SMEM_ADDR_Q3 ((float * const) 0xff00c000)
// #define SPAD_ADDR_Q0 0x0
// #define SPAD_ADDR_Q1 0x200
// #define SPAD_ADDR_Q2 0x400
// #define SPAD_ADDR_Q3 0x600
// #define NUM_THREADS_IN_CLUSTER 256

// fp32 4x4
// #define TILE_M 32
// #define TILE_N 32
// #define TILE_K 32
// #define SMEM_ADDR_Q0 ((float * const) 0xff000000)
// #define SMEM_ADDR_Q1 ((float * const) 0xff001000)
// #define SMEM_ADDR_Q2 ((float * const) 0xff002000)
// #define SMEM_ADDR_Q3 ((float * const) 0xff003000)
// #define SPAD_ADDR_Q0 0x0
// #define SPAD_ADDR_Q1 0x80
// #define SPAD_ADDR_Q2 0x100
// #define SPAD_ADDR_Q3 0x180
// #define NUM_THREADS_IN_CLUSTER 256

#define NUM_CLUSTERS 1
// (NUM_CORES * NUM_WARPS * NUM_THREADS)

#define rd_cycles_force(x) asm volatile ("csrr %0, mcycle" : "=r" (x))
#define rd_cycles(x) rd_cycles_force(x)
#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
#define PRINTF(...) sprintf(PRINT_BUF, __VA_ARGS__)
// #define PRINTF(...) vx_printf(__VA_ARGS__)
#define SWISH(beta, x) ((x) / (1 + exp(-(beta) * (x))))
#define POWER

typedef uint16_t smem_elem_t;
// typedef float smem_elem_t;

inline void threadblock_barrier(unsigned int barrier_id, unsigned int count) __attribute__((convergent)) {
  vx_fence();
  vx_barrier(barrier_id, count);
}

void thread_block_matmul_gemmini(kernel_arg_t *__UNIFORM__ arg,
                                 const uint32_t threadblock_id,
                                 const uint32_t tid_in_threadblock) {
  asm volatile ("matmul_start_%=:" :: );
  const smem_elem_t * const A = (const smem_elem_t * const) arg->addr_a;
  const smem_elem_t * const B = (const smem_elem_t * const) arg->addr_b;
  smem_elem_t * const C = (smem_elem_t * const) arg->addr_c;

  if (HW_TID() == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    #ifndef POWER
    PRINTF("start\n");
    #endif
  }

  vx_fence();

  uint32_t marker0, marker1;
  rd_cycles_force(marker0);
  MARK_BEG();

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;
  const uint32_t num_tiles_m = dim_m / TILE_M;
  const uint32_t num_tiles_n = dim_n / TILE_N;
  const uint32_t num_tiles_k = dim_k / TILE_K;
  constexpr uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;

  const uint32_t num_tile_rows_per_tb = num_tiles_m / NUM_CLUSTERS;

  if (HW_TID() == 0) {
    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);
    // gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);

    for (uint32_t tile_i = num_tile_rows_per_tb * threadblock_id;
                  tile_i < num_tile_rows_per_tb * (threadblock_id + 1);
                  tile_i += 1) {
      for (uint32_t tile_j = 0; tile_j < num_tiles_n; tile_j += 1) {
        for (uint32_t tile_k = 0; tile_k < num_tiles_k; tile_k += 1) {
          uint32_t a_hexadecile = (tile_k & 1) << 2;
          uint32_t b_hexadecile = a_hexadecile + 11;
          gemmini_tile_load_ab(A, B,
              a_hexadecile, b_hexadecile, tile_i, tile_j, tile_k,
              dim_m, dim_n, dim_k, TILE_M, TILE_N, TILE_K);
          /* DO STUFF */
          gemmini_fence();
          gemmini_tile_compute</*store_to_spad=*/false>(
              a_hexadecile, b_hexadecile, 0 /*d_hexadecile*/, tile_k > 0);
        }

        /*
        gemmini_fence();
        gemmini_tile_store_c_spad(a_hexadecile); // then activate in spad
        */
        gemmini_fence();
        gemmini_tile_store_c_gmem(C, tile_i, tile_j, dim_m, dim_n, TILE_M, TILE_N);
      }
    }

    MARK_END();
    rd_cycles_force(marker1);

    #ifndef POWER
      if (HW_TID() == 0) {
        PRINTF("\ncomplete\n");
        PRINTF("total cycles:         %d\n", marker1 - marker0);
        for (int i = 0; i < dim_m; i += 8) {
          for (int j = 0; j < dim_n; j += 8) {
            PRINTF("%04x %04x ", (int) (C[i * dim_n + j]), (int) (C[i * dim_n + j + 4]));
          }
          PRINTF("\n");
        }
      }
    #endif
  } else {
    if (HW_TID() > 8) {
      asm volatile("li x1, 0xa0a0a0a0");
      asm volatile("li x2, 0xa0a0a0a0");
      asm volatile("li x3, 0xa0a0a0a0");
      asm volatile("li x4, 0xa0a0a0a0");
      asm volatile("li x5, 0xa0a0a0a0");
      asm volatile("li x6, 0xa0a0a0a0");
      asm volatile("li x7, 0xa0a0a0a0");
      asm volatile("li x8, 0xa0a0a0a0");
      asm volatile("li x9, 0xa0a0a0a0");
      asm volatile("li x10, 0xa0a0a0a0");
      asm volatile("li x11, 0xa0a0a0a0");
      asm volatile("li x12, 0xa0a0a0a0");
      asm volatile("li x13, 0xa0a0a0a0");
      asm volatile("li x14, 0xa0a0a0a0");
      asm volatile("li x15, 0xa0a0a0a0");
      asm volatile("li x16, 0xa0a0a0a0");
      asm volatile("li x17, 0xa0a0a0a0");
      asm volatile("li x18, 0xa0a0a0a0");
      asm volatile("li x19, 0xa0a0a0a0");
      asm volatile("li x20, 0xa0a0a0a0");
      asm volatile("li x21, 0xa0a0a0a0");
      asm volatile("li x22, 0xa0a0a0a0");
      asm volatile("li x23, 0xa0a0a0a0");
      asm volatile("li x24, 0xa0a0a0a0");
      asm volatile("li x25, 0xa0a0a0a0");
      asm volatile("li x26, 0xa0a0a0a0");
      asm volatile("li x27, 0xa0a0a0a0");
      asm volatile("li x28, 0xa0a0a0a0");
      asm volatile("li x29, 0xa0a0a0a0");
      asm volatile("li x30, 0xa0a0a0a0");
      asm volatile("li x31, 0xa0a0a0a0");
      asm volatile("vx_tmc zero");
    }
  }
  vx_fence();
  // threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);
  vx_tmc(0);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  const int threadblock_id = task_id / NUM_THREADS_IN_CLUSTER;
  const int tid_in_threadblock = task_id % NUM_THREADS_IN_CLUSTER;

  thread_block_matmul_gemmini(arg, threadblock_id, tid_in_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;
  const uint32_t grid_size = num_threads_in_cluster * NUM_CLUSTERS;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
