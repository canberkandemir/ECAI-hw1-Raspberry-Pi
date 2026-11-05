# ECAI27 — Homework 1: Sparse GEMM
## Development Log

**Course:** Efficient Computing for Artificial Intelligence  
**Project:** HW1 — 4×4 Block-Sparse GEMM Optimization  
**Hardware:** Raspberry Pi 4  
**Programming Language:** C  
**Development Environment:** Visual Studio Code, Linux, Git, GitHub  
**Team:** Egehan and Canberk  
**Submission Deadline:** 2 December 2026, 23:59

---

## Session 01 — 5 November 2026
### Project Initialization and Documentation Setup

**Objective:** Understand the homework requirements, establish the shared development environment, and prepare the project documentation.

### 1. Assignment Requirements

The homework requires implementing and benchmarking four optimized versions of 4×4 block-sparse general matrix multiplication (GEMM):

1. **NEON:** Use four FP32 NEON lanes to update four adjacent output columns.
2. **Four-row tiled:** Process four input rows together to reuse weight blocks, without NEON.
3. **OpenMP:** Parallelize independent output rows using OpenMP, without tiling or NEON.
4. **Combined:** Integrate four-row tiling, NEON SIMD, and OpenMP parallelism.

The fixed problem configuration is:

- M = 128
- K = 512
- N = 512
- Number of nonzero 4×4 blocks = 4096

The provided scalar and dense implementations serve as correctness and performance baselines.

### 2. Development Environment

The project uses a Raspberry Pi 4 with Linux and Visual Studio Code.

A shared GitHub repository has been configured:

https://github.com/canberkandemir/ECAI-hw1-Raspberry-Pi

The local Git repository is located at:

`~/workdir/ECAI-hw1-Raspberry-Pi/`

The original starter files remain at:

`~/workdir/hw1/`

A working copy of the starter files has been copied into the shared repository directory under `hw1/`.

### 3. Project Structure

The starter code includes:

- `include/`: Shared header files.
- `src/sparse_gemm/`: Baseline implementations, student kernel templates, tests and benchmark code.
- `Makefile`: Build configuration.

Additional directories created for development documentation:

- `docs/`: Technical notes and development history.
- `results/raw/`: Raw benchmark measurements.
- `results/analysis/`: Processed results, summaries and figures.

### 4. Completed Activities

- Reviewed the official assignment instructions.
- Reviewed the required LaTeX report format.
- Confirmed the GitHub remote configuration.
- Verified that the remote repository has no initial commit.
- Located the original homework starter files.
- Copied the starter files into the local Git repository.
- Verified the expected source files, headers and Makefile.
- Created documentation and results directories.
- Created the initial development log file.

### 5. Testing and Implementation Status

No C kernel implementations were modified during this session.

No compilation, correctness tests or performance benchmarks were executed during this session.

Git changes have not yet been committed or pushed to the remote repository.

### 6. Next Actions

- Inspect the starter code and Makefile.
- Verify the Git branch and working-tree status.
- Add the official LaTeX report template.
- Establish the initial GitHub commit and collaboration workflow.
- Compile and test the baseline implementations.
- Record baseline benchmark measurements.

### Session Status

Documentation setup in progress. No kernel implementation work started.


---

## Session 02 — 5 November 2026
### Exercise 1: Baseline Validation and Benchmarking

**Developer:** Canberk  
**Hardware:** Raspberry Pi 4  
**Status:** Completed

### 1. Objective

Validate the provided sparse scalar GEMM against the dense reference and benchmark the three baseline implementations:

- `dense_scalar`
- `dense_tiled_neon_openmp_TN8_T4`
- `sparse_scalar`

### 2. Fixed Test Configuration

- M = 128
- K = 512
- N = 512
- Nonzero 4×4 blocks (nnzb) = 4096
- Density = 0.25
- Data type = FP32
- Compiler optimization = `-O0`
- Target CPU = Cortex-A72

### 3. Correctness Validation

The provided test program was compiled and executed on the Raspberry Pi 4.

Results:

- `dense_tiled_neon_openmp_TN8_T4`: PASS
- `sparse_scalar`: PASS
- Optimized sparse kernels: SKIP (not implemented yet)

The sparse scalar implementation passed correctness validation against the dense reference.

No GEMM kernel implementations were modified.

### 4. Benchmark Methodology

The provided benchmark program was used with:

- 2 warmup executions
- 5 measured timing samples per kernel
- Monotonic clock timing
- Automatic selection of repetitions
- Correctness validation before timing

The professor's original `benchmark.c` formatting and logic were preserved.

Only the following lines were added after the timing sample calculation:

```c
        // Added for Exercise 1: print individual timing samples.
        fprintf(stderr, "SAMPLE,%s,%zu,%.6f\n",
                item->name, sample + 1, samples[sample]);
```

This addition prints individual execution times for calculating latency standard deviation. It does not modify GEMM computation or timing methodology.

### 5. Final Benchmark Results

Benchmark run ID: `20261105T174226`

| Kernel | Threads | Median (ms) | Sample Std. (ms) |
|---|---:|---:|---:|
| Dense scalar | 1 | 590.5 | 6.8 |
| Dense tiled NEON OpenMP | 4 | 85.7 | 4.0 |
| Sparse scalar | 1 | 98.1 | 0.9 |

Latency values are rounded to one decimal place.

Standard deviation was calculated externally from five individual timing samples using the sample standard deviation formula (n-1).

### 6. Analytical Operational Intensity

Under the ideal single-level-memory model with unlimited registers and unique data transfers:

- Dense scalar: 42.67 FLOP/byte
- Dense tiled NEON OpenMP: 42.67 FLOP/byte
- Sparse scalar: 20.88 FLOP/byte

Sparse OI includes input/output matrices, stored FP32 weights, and BCC metadata with 32-bit indices.

### 7. Performance Analysis

The fastest dense baseline was `dense_tiled_neon_openmp_TN8_T4`.

- Dense optimized was approximately 6.9× faster than dense scalar.
- Sparse scalar was approximately 6.0× faster than dense scalar.
- Dense optimized was approximately 1.14× faster than sparse scalar.

Although sparse scalar reduces arithmetic by skipping zero blocks, the optimized dense implementation benefits from tiling, NEON SIMD, and four-thread OpenMP parallelism.

### 8. Generated Files

- `results/raw/results_ex1_final.csv`
- `results/raw/samples_ex1_final.log`

The CSV contains the benchmark summary. The log contains individual timing samples and benchmark diagnostic messages.

### 9. Completion Status

Exercise 1 baseline validation and benchmarking completed.

The final LaTeX report still needs to be updated with these measurements.

Exercises 2–5 remain to be implemented and tested.
