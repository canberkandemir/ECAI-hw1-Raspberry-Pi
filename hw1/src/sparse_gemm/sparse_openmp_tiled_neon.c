
#include "sparse_gemm.h"
#include <arm_neon.h>

int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t tile = 0; tile < (m + 3) / 4; ++tile) {

        // Start of the current 4-row tile
        size_t i0 = 4 * tile;
        size_t rows = (m - i0 < 4) ? (m - i0) : 4;

        // Process four output columns at a time
        for (size_t g = 0; g < w->n / 4; ++g) {

            // One NEON accumulator for each output row
            float32x4_t acc[4];

            for (size_t t = 0; t < rows; ++t)
                acc[t] = vdupq_n_f32(0.0f);

            // Iterate through stored nonzero blocks
            for (size_t p = w->block_col_ptr[g];
                 p < w->block_col_ptr[g + 1]; ++p) {

                size_t k0 = 4 * (size_t)w->block_row_idx[p];

                for (size_t r = 0; r < 4; ++r) {

                    // Load four weights once
                    float32x4_t weights = vld1q_f32(
                        &w->block_values[16 * p + 4 * r]
                    );

                    // Reuse weights across up to four rows
                    for (size_t t = 0; t < rows; ++t) {

                        float av = a[(i0 + t) * w->k + k0 + r];

                        float32x4_t a_vec = vdupq_n_f32(av);

                        acc[t] = vfmaq_f32(
                            acc[t], a_vec, weights
                        );
                    }
                }
            }

            // Store the completed output tile
            for (size_t t = 0; t < rows; ++t)
                vst1q_f32(
                    &c[(i0 + t) * w->n + 4 * g],
                    acc[t]
                );
        }
    }

    return SPARSE_OK;
}
