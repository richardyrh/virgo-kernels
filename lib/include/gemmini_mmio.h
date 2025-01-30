#ifndef GEMMINI_MMIO_H
#define GEMMINI_MMIO_H
#ifndef GEMMINI_PARAMS_H
#error INCLUDE GEMMINI.H FIRST
#endif

/* shared memory constants and helpers */
/* =================================== */
#define SMEM_BASE 0xff000000
// 16KB
// #define SMEM_SIZE 0x4000
// 64KB
// #define SMEM_SIZE 0x10000
// 128KB (FP16 GEMM)
// #define SMEM_SIZE 0x20000
// 256KB (FlashAttention)
#define SMEM_SIZE 0x40000

#define SMEM_MASK (SMEM_SIZE - 1)
#define SMEM_ADDR_END (SMEM_BASE + SMEM_SIZE)

#define SPAD_BASE 0x0
#define SPAD_ROW_SIZE (DIM * sizeof(elem_t))
#define SPAD_NUM_ROWS (SMEM_SIZE / SPAD_ROW_SIZE)
#define SPAD_MASK (SPAD_NUM_ROWS - 1)

#define PRINT_BUF ((char *) (SMEM_ADDR_END))
#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
#define SMEM_TO_SPAD(smem_addr) (SPAD_BASE + ((smem_addr) & SMEM_MASK) / SPAD_ROW_SIZE)
#define SPAD_TO_SMEM(spad_addr) (SMEM_BASE + ((spad_addr) & SPAD_MASK) * SPAD_ROW_SIZE)

// convert normal matrix i,j into tiled smem offset
// top_in_tiles = i / DIM
// left_in_tiles = j / DIM
// num_tiles_before_current = top_in_tiles * (J / DIM) + left_in_tiles
// smem_addr = num_tiles_before_current * DIM * DIM + (i % DIM) * DIM + (j % DIM)
#define SMEM_MAT_OFFSET(i, j, J) \
    (((i) / DIM * (J) / DIM + (j) / DIM) * DIM * DIM + ((i) % DIM) * DIM + ((j) % DIM))

/* gemmini mmio interface */
/* ====================== */
static size_t gemmini_tile_idx[NUM_THREADS * NUM_WARPS * NUM_CORES * NUM_CLUSTERS] = {0};
#define use_gemmini(i) {gemmini_tile_idx[HW_TID()] = (i);}
#define GEMMINI_TILE_IDX() (gemmini_tile_idx[HW_TID()])
#define GEMMINI_CISC_IMM(x, i) ((x) + 32 * (i))
#define GEMMINI_CTRL (SMEM_BASE + SMEM_SIZE + 0x3000 + 0x100 * GEMMINI_TILE_IDX())
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)
#define GEMMINI_OCCUPANCY_ADDR (GEMMINI_CTRL + 0x28)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}

/* additional intrinsics */
/* ===================== */
#define loop_matmul_skips(skip_lda, skip_ldb, skip_ldd, skip_ex, skip_stc) \
  (((skip_lda) | ((skip_ldb) << 1) | ((skip_ldd) << 2) | ((skip_ex) << 3) | ((skip_stc) << 4)) << 3)

#define sp_tiled_matmul_full_spad_ws(A_sp_addr_start, B_sp_addr_start, D_sp_addr_start, C_dst_sp_addr_start,\
  I, J, K, pad_I, pad_J, pad_K, a_transpose, b_transpose, full_C, low_D, acc, act, skips) \
  gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A_sp_addr_start, (B_sp_addr_start) + (K) * (J) * DIM, NULL, \
  C_dst_sp_addr_start, a_transpose, b_transpose, full_C, low_D, acc, act, 0, 0, false, skips)

#define gemmini_status() ({uint32_t status; asm volatile ("csrr %0, 0xacc" : "=r" (status)); status;})

#undef gemmini_fence
//#define gemmini_fence() { while (gemmini_status()); }
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }

#define virgo_fence(n) { while (*((volatile uint32_t *) GEMMINI_OCCUPANCY_ADDR) > n) asm volatile ("nop"); }

/* cisc instructions */
/* ================= */

// bits [4:0] is the opcode
// bits [7:5] is the target gemmini id, zero-indexed
// #define GEMMINI_CISC_CMD_I(x) asm("csrwi 0xacc, %0" :: "i" (x))
#define GEMMINI_CISC_CMD_I(x) asm("csrw 0xacc, %0" :: "r" (x)) // use registers even for immediate calls for now
#define GEMMINI_CISC_CMD_R(x) asm("csrw 0xacc, %0" :: "r" (x))

#define GEMMINI_CISC_COMPUTE_HEXADECILES 0
#define GEMMINI_CISC_COMPUTE_AND_STORE_TO_SPAD 1
#define GEMMINI_CISC_MANUAL 2
#define GEMMINI_CISC_SET_AB_STRIDE 8
#define GEMMINI_CISC_STORE_TO_SPAD 9
#define GEMMINI_CISC_LOAD_TO_HEXADECILES 10
#define GEMMINI_CISC_SET_DC_STRIDE 11
#define GEMMINI_CISC_STORE_TO_GMEM 12

/* high level virgo routines */
/* ========================= */
inline void gemmini_tile_load_ab(const elem_t * const a_addr, const elem_t * const b_addr,
    const uint32_t a_hexadecile, const uint32_t b_hexadecile,
    const uint32_t tile_idx_i, const uint32_t tile_idx_j, const uint32_t tile_idx_k,
    const uint32_t mat_size_m, const uint32_t mat_size_n, const uint32_t mat_size_k,
    const uint32_t tile_size_m, const uint32_t tile_size_n, const uint32_t tile_size_k) {

  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
      (uint64_t) (a_addr + tile_idx_i * tile_size_m * mat_size_k + tile_idx_k * tile_size_k),
      (uint64_t) (b_addr + tile_idx_k * tile_size_k * mat_size_n + tile_idx_j * tile_size_n), k_LOOP_WS_CONFIG_ADDRS_AB)
  GEMMINI_CISC_CMD_R((mat_size_n << 20) | (mat_size_k << 8) | GEMMINI_CISC_SET_AB_STRIDE);
  GEMMINI_CISC_CMD_R((b_hexadecile << 16) | (a_hexadecile << 8) | GEMMINI_CISC_LOAD_TO_HEXADECILES);
}

template <bool store_to_spad = false>
inline void gemmini_tile_compute(const uint32_t a_hexadecile,
                                 const uint32_t b_hexadecile,
                                 const uint32_t d_hexadecile,
                                 const bool accumulate) {
  if constexpr (!store_to_spad) {
    GEMMINI_CISC_CMD_R((static_cast<uint32_t>(accumulate) << 24) |
                       (b_hexadecile << 16) | (a_hexadecile << 8) |
                       GEMMINI_CISC_COMPUTE_HEXADECILES);
  } else {
    GEMMINI_CISC_CMD_R((d_hexadecile << 24) | (b_hexadecile << 16) |
                       (a_hexadecile << 8) | GEMMINI_CISC_COMPUTE_AND_STORE_TO_SPAD);
  }
}

inline void gemmini_tile_store_c_gmem(elem_t * const c_addr,
    const uint32_t tile_idx_i, const uint32_t tile_idx_j,
    const uint32_t mat_size_m, const uint32_t mat_size_n,
    const uint32_t tile_size_m, const uint32_t tile_size_n) {

  elem_t * const dram_c_tile_start = c_addr + tile_idx_i * tile_size_m * mat_size_n + tile_idx_j * tile_size_n;
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, (uint64_t) dram_c_tile_start, k_LOOP_WS_CONFIG_ADDRS_DC)

  GEMMINI_CISC_CMD_R((mat_size_n << 20) | GEMMINI_CISC_SET_DC_STRIDE);
  GEMMINI_CISC_CMD_I(GEMMINI_CISC_STORE_TO_GMEM);

  // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST, k_LOOP_WS_CONFIG_BOUNDS)
  // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, mat_size_n, k_LOOP_WS_CONFIG_STRIDES_DC)
  // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, loop_matmul_skips(1, 1, 1, 1, 0), k_LOOP_WS)
}

inline void gemmini_tile_store_c_spad(const uint32_t c_hexadecile) {
  GEMMINI_CISC_CMD_R(((uint32_t) (c_hexadecile << 8)) | GEMMINI_CISC_STORE_TO_SPAD);
}

inline void gemmini_manual_job() {
  GEMMINI_CISC_CMD_I(GEMMINI_CISC_MANUAL);
}

/* inline static void sp_tiled_matmul_full_spad_ws(const uint32_t A_sp_addr_start, const uint32_t B_sp_addr_start,
                                                const uint32_t D_sp_addr_start, const uint32_t C_dst_sp_addr_start,
                                                size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
                                                bool a_transpose, bool b_transpose,
                                                bool full_C, bool low_D, bool acc,
                                                int act, int skip_mvout) {

  gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K,
                       A_sp_addr_start, B_sp_addr_start + K * J * DIM, NULL, C_dst_sp_addr_start,
                       a_transpose, b_transpose,
                       full_C, low_D, acc,
                       act, 0, 0, false, skip_mvout); */
  /*
  return;


  // const uint32_t A_sp_addr_start = 0;
  // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  // const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 2 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  // const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
  //   (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = 1; //full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  // const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
  gemmini_fence();

  if (a_transpose || b_transpose || (I < 4)) {
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < J; j++) {
        for (size_t i = 0; i < I; i++) {
          const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
            (A_sp_addr_start + (i*K + k)*DIM);
          const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
            (B_sp_addr_start + (k*J + j)*DIM);
          const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
          // Compute
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr | ((k == 0 ? 0 : 1) << (ADDR_LEN-2));
          gemmini_extended_preload(pre_sp_addr, out_sp_addr, DIM, DIM, DIM, DIM);
          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          }
          if (k == K - 1) {
            // Move-out C (if not normalizing)
            // if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
              const size_t rounded_j = j; // (j / C_blocks) * C_blocks;
              const uint32_t rounded_C_sp_addr = C_sp_addr; // C_sp_addr_start + (i*J + rounded_j)*DIM;

              const uint32_t C_dst_sp_addr = ((uint32_t) C_dst_sp_addr_start) + (i * J + rounded_j) * DIM; // * DIM * sizeof_C;

              // const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
              constexpr size_t cols = DIM; // blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
              constexpr size_t rows = DIM; // DIM - (i == I - 1 ? pad_I : 0);

              gemmini_extended_mvout_spad(C_dst_sp_addr, 1, rounded_C_sp_addr, cols, rows);
            // }
          }
        }
      }
    }
  } else {
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < J; j++) {
        uint32_t A_sp_addr = A_sp_addr_start + k * DIM; // (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
        uint32_t C_sp_addr = C_sp_addr_start + j * DIM; // (i*J + j)*DIM;
        for (size_t i = 0; i < I; i += 4) {
          // Compute
          // constexpr uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          const uint32_t out_sp_addr = C_sp_addr | ((k == 0 ? 0 : 1) << (ADDR_LEN-2));
          if (i == 0) { // First iteration
            gemmini_extended_preload(B_sp_addr, out_sp_addr, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 2 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 2 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 3 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 3 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 2 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 2 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
            gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr + 3 * J * DIM, DIM, DIM, DIM, DIM);
            gemmini_extended_compute_accumulated(A_sp_addr + 3 * K * DIM, GARBAGE_ADDR, DIM, DIM, DIM, DIM);
          }
          if (k == K - 1) {
            for (int x = 0; x < 3; x++) gemmini_fence();
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + (i * J + j) * DIM, 1, C_sp_addr, DIM, DIM);
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + ((i + 1) * J + j) * DIM, 1, C_sp_addr + J * DIM, DIM, DIM);
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + ((i + 2) * J + j) * DIM, 1, C_sp_addr + 2 * J * DIM, DIM, DIM);
            gemmini_extended_mvout_spad((uint32_t) C_dst_sp_addr_start + ((i + 3) * J + j) * DIM, 1, C_sp_addr + 3 * J * DIM, DIM, DIM);
          }
          A_sp_addr += 4 * K * DIM;
          C_sp_addr += 4 * J * DIM;
        }
      }
    }
  }
  gemmini_fence();
}*/


#endif
