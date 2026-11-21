import csv
from pathlib import Path

# Fixed homework configuration
M = 128
K = 512
N = 512
NNZB = 4096
BLOCK_SIZE = 4
FP32_BYTES = 4
INDEX_BYTES = 4

# Dense GEMM
dense_flops = 2 * M * K * N

dense_bytes = FP32_BYTES * (
    M * K +
    K * N +
    M * N
)

dense_oi = dense_flops / dense_bytes

# Sparse GEMM (4x4 Block Compressed Column)
sparse_flops = 2 * M * NNZB * (BLOCK_SIZE ** 2)

input_bytes = M * K * FP32_BYTES
output_bytes = M * N * FP32_BYTES
block_values_bytes = NNZB * (BLOCK_SIZE ** 2) * FP32_BYTES
block_row_idx_bytes = NNZB * INDEX_BYTES
block_col_ptr_bytes = (N // BLOCK_SIZE + 1) * INDEX_BYTES

sparse_bytes = (
    input_bytes +
    output_bytes +
    block_values_bytes +
    block_row_idx_bytes +
    block_col_ptr_bytes
)

sparse_oi = sparse_flops / sparse_bytes

# Display results
print("ECAI HW1 - Analytical Operational Intensity")
print("-----------------------------------------")
print(f"Dense FLOPs: {dense_flops:,}")
print(f"Dense bytes: {dense_bytes:,}")
print(f"Dense OI: {dense_oi:.2f} FLOP/byte")
print()
print(f"Sparse FLOPs: {sparse_flops:,}")
print(f"Sparse bytes: {sparse_bytes:,}")
print(f"Sparse OI: {sparse_oi:.2f} FLOP/byte")

# Save results to CSV
output_path = Path(__file__).parent / "oi_results.csv"

with output_path.open("w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["kernel_type", "flops", "bytes", "oi_flop_per_byte"])
    writer.writerow(["dense", dense_flops, dense_bytes, f"{dense_oi:.6f}"])
    writer.writerow(["sparse", sparse_flops, sparse_bytes, f"{sparse_oi:.6f}"])

print()
print(f"Results saved to: {output_path}")