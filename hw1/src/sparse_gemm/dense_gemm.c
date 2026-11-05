#include "sparse_gemm.h"

#include <arm_neon.h>

// By using this scalar GEMM implementation with the i → j → q loop order, 
// we gain simplicity and low implementation complexity, 
// but we sacrifice some performance, especially for large matrices.

//C. Limited data reuse
//When calculating C[0][0], we traverse the first row of A.

//When calculating C[0][1], we traverse that same row again.

void gemm_dense_scalar(const float *a, const float *w, float *c,
                       size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t q = 0; q < k; ++q)
                sum += a[i * k + q] * w[q * n + j];
            c[i * n + j] = sum;
        }
}

// Unlike scalar GEMM, which computes C[0][0], then C[0][1], and so on,
// this implementation computes an entire group of 8 output elements together.
//For A: We move column by column within a fixed row, taking one element at a time.
//For W: We move row by row, taking 8 consecutive elements (columns) at a time.
// The biggest improvement is that we reuse the same value of A across 8 output columns.
// Less flexibility for matrix dimensions
// Your implementation processes 8 columns at a time:
// If n = 8 or n = 16, that works.
// But if n = 10, the second tile attempts to process columns 8–15, even though columns 10–15 do not exist.
// This can cause out-of-bounds reads and writes.
// To support arbitrary dimensions, we need remainder handling, such as a scalar cleanup loop.

void gemm_dense_tiled_neon_openmp_TN8_T4(const float *a, const float *w,
                                         float *c, size_t m, size_t k,
                                         size_t n)
{
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) //The outer loop iterates through the rows of A and C. 
    {
        for (size_t j = 0; j < n; j += 8) //Process 8 columns at a time
          {
            float32x4_t acc0 = vdupq_n_f32(0.0f); // acc0 will calculate columns 0–3.
            float32x4_t acc1 = vdupq_n_f32(0.0f); // acc1 will calculate columns 4–7.
            for (size_t q = 0; q < k; ++q) {
                float32x4_t av = vdupq_n_f32(a[i * k + q]); // Broadcast one value from A
                acc0 = vfmaq_f32(acc0, av, vld1q_f32(w + q * n + j)); // acc0 maintains the running sums for columns 0–3.
                acc1 = vfmaq_f32(acc1, av, vld1q_f32(w + q * n + j + 4)); // acc1 maintains the running sums for columns 4–7.
            }
            vst1q_f32(c + i * n + j, acc0);
            vst1q_f32(c + i * n + j + 4, acc1);
        }
    }
}
