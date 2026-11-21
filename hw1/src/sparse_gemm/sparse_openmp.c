
#include "sparse_gemm.h"

int gemm_sparse_openmp(const float *a, const sparse_weights *w,
                       float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;

    // Parallelize independent output rows.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m; ++i) {

        // Each group corresponds to four output columns.
        for (size_t g = 0; g < w->n / 4; ++g) {

            // Local accumulators for this output row and group.
            float acc[4] = {0};

            // Traverse only retained BCC blocks.
            for (size_t p = w->block_col_ptr[g];
                 p < w->block_col_ptr[g + 1]; ++p) {

                size_t k0 = 4 * (size_t)w->block_row_idx[p];

                for (size_t r = 0; r < 4; ++r) {
                    float av = a[i * w->k + k0 + r];

                    for (size_t col = 0; col < 4; ++col) {
                        acc[col] += av *
                            w->block_values[16 * p + 4 * r + col];
                    }
                }
            }

            // Store the four output values.
            for (size_t col = 0; col < 4; ++col) {
                c[i * w->n + 4 * g + col] = acc[col];
            }
        }
    }

    return SPARSE_OK;
}
