
#include "sparse_gemm.h"

int gemm_sparse_tiled(const float *a, const sparse_weights *w, float *c,
                      size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;

    // Process the input matrix in tiles of four rows.
    for (size_t i = 0; i < m; i += 4) {

        // Each BCC group corresponds to four output columns.
        for (size_t g = 0; g < w->n / 4; ++g) {

            // Four input rows x four output columns.
            float acc[4][4] = {0};

            // Visit only the stored nonzero weight blocks.
            for (size_t p = w->block_col_ptr[g];
                 p < w->block_col_ptr[g + 1]; ++p) {

                size_t k0 = 4 * (size_t)w->block_row_idx[p];

                // Iterate over the 4x4 stored weight block.
                for (size_t r = 0; r < 4; ++r) {

                    // Load one input value from each of the four rows.
                    float a0 = a[(i + 0) * w->k + k0 + r];
                    float a1 = a[(i + 1) * w->k + k0 + r];
                    float a2 = a[(i + 2) * w->k + k0 + r];
                    float a3 = a[(i + 3) * w->k + k0 + r];

                    // Process four output columns.
                    for (size_t col = 0; col < 4; ++col) {

                        // Load the weight once.
                        float weight =
                            w->block_values[16 * p + 4 * r + col];

                        // Reuse the weight across all four input rows.
                        acc[0][col] += a0 * weight;
                        acc[1][col] += a1 * weight;
                        acc[2][col] += a2 * weight;
                        acc[3][col] += a3 * weight;
                    }
                }

            }

            // Write the accumulated output tile.
            for (size_t t = 0; t < 4; ++t) {
                for (size_t col = 0; col < 4; ++col) {
                    c[(i + t) * w->n + 4 * g + col] =
                        acc[t][col];
                }
            }
        }
    }

    return SPARSE_OK;
}
