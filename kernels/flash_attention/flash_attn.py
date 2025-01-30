import sys
import numpy as np

def parse_mnk():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} dimM dimN dimK", file=sys.stderr)
        sys.exit(1)
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    return (m, n, k)


# Reorder array in a way that groups two adjacent elements along the column to
# be now adjacent along the row.  This way, when the resulting fp16 array is
# read in column-major order with 32-bit granularity, the fp16 elements will be
# read in the same order as regular fp32 elements in column-major.
#
# For example:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# becomes
# [[1 3 2 4]
#  [5 7 6 8]]
def pack_fp16_by_column(array):
    rows = array.shape[0]
    cols = array.shape[1]

    T = array.transpose([1, 0])
    T_packed = T.reshape([cols, -1, 2])
    result = T_packed.transpose([1, 0, 2])
    return result


# Do the same as pack_fp16_by_column, but for every two elements along the row.
def pack_fp16_by_row(array):
    rows = array.shape[0]
    cols = array.shape[1]

    result = array.reshape([rows, -1, 2])
    return result


if __name__ == "__main__":
    seqlen, _, headdim = parse_mnk()

    rand = True
    if not rand:
        A_array = np.arange(seqlen * headdim).reshape([seqlen, headdim])
        B_array = np.arange(headdim * seqlen).reshape([headdim, seqlen])
        C_array = np.arange(seqlen * seqlen).reshape([seqlen, headdim])
    else:
        np.random.seed(0)
        A_array = np.random.rand(seqlen, headdim) - 0.5
        B_array = np.random.rand(headdim, seqlen) - 0.5
        C_array = np.random.rand(seqlen, headdim) - 0.5
        # C_array = np.zeros([M, N])

    fp16 = False
    if fp16:
        A_packed = pack_fp16_by_row(A_array)
        AT_packed = A_packed.transpose([1, 0, 2])
        AT_array = AT_packed.reshape([-1, seqlen * 2])
        AT_array.astype('float16').tofile("input.a.col.bin")
        # print('AT:')
        # print(AT_array)
        B_packed = pack_fp16_by_column(B_array)
        B_array = B_packed.reshape([-1, headdim * 2])
        B_array.astype('float16').tofile("input.b.row.bin")
        # print('B:')
        # print(B_array)
    else:
        A_array.astype('float32').tofile("input.a.row.bin")
        AT_array = A_array.transpose([1, 0])
        AT_array.astype('float32').tofile("input.a.col.bin")
        B_array.astype('float32').tofile("input.b.bin")
        C_array.astype('float32').tofile("input.c.bin")
        # print('AT:')
        # print(AT_array)
        # print('B:')
        # print(B_array)

    assert((seqlen % 64) == 0)

    Br = 64
    Bc = Br

    rowmax = np.zeros([Br])
    rowsum = np.zeros([Br])
    O = np.zeros([Br, headdim])

    def exp2(x):
        return (x**2) / 2.0 + x + 1.0

    full_S = A_array @ B_array
    full_S_T = full_S.transpose([1, 0])
    full_S.astype('float32').tofile("full_S.bin")

    col_to_save = 0

    for col in range(0, seqlen, Bc):
        print(f"tile iteration {col}~{col + Bc} ======================================")

        # FIXME: only work with the first 64 rows of Q for now
        Q_tile = A_array[0:64, :]
        K_tile = B_array[:, col:col+Bc]

        S = Q_tile @ K_tile
        if col == col_to_save:
            print('S_expected:')
            print(S)
            S.astype('float32').tofile("S_expected.bin")

        # generate rowmax result in online softmax
        rowmax_this = np.max(S, axis=1)
        rowmax_prev = rowmax.copy()
        rowmax = np.maximum(rowmax, rowmax_this)
        if col == col_to_save:
            rowmax.astype('float32').tofile("rowmax.bin")

        # subtrace rowmax from each row by broadcasting
        # (placeholder for exp)
        x = S - rowmax[:, np.newaxis]
        P = exp2(x)
        # for i in range(3, 4):
        #     P += (x**i) / np.math.factorial(i)
        # P = np.exp(exp)
        # print('P error:')
        # print(P / np.exp(x))
        if col == col_to_save:
            print('P_expected:')
            print(P)
            P.astype('float32').tofile("P_expected.bin")
            P.transpose([1, 0]).astype('float32').tofile("P_expected.col.bin")

        rowsum_this = np.sum(P, axis=1)
        x = rowmax_prev - rowmax_this
        rowsum = exp2(x) * rowsum + rowsum_this
        if col == col_to_save:
            rowsum.astype('float32').tofile("rowsum.bin")

        x = rowmax_prev - rowmax
        O = O / (exp2(x)[:, np.newaxis])
        if col == col_to_save:
            print('O_before_PV:')
            print(O)
            O.astype('float32').tofile("O_before_PV.bin")

        V = C_array[col:col+Bc, :]
        if col == col_to_save:
            V.astype('float32').tofile("V_expected.bin")
        # O = P.transpose([1, 0]) @ V
        O = O + P @ V
        if col == col_to_save:
            print('O_after_PV:')
            print(O)
            O.astype('float32').tofile("O_after_PV.bin")
