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