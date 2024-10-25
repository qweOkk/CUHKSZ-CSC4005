//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstring>
Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);
    size_t M_align=((M-1)/8)*8+8;
    size_t N_align=((N-1)/8)*8+8;
    size_t K_align=((K-1)/8)*8+8;
    // Initialize the result matrix
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result[i][j] = 0;
        }
    }

    // Allocate and initialize one-dimensional arrays for m1 and m2
    int* m1 = (int*)_mm_malloc(M * K_align * sizeof(int), 32); 
    int* m2 = (int*)_mm_malloc(K * N_align * sizeof(int), 32);
    int* res = (int*)_mm_malloc(M * N_align * sizeof(int), 32);
    memset(res, 0, M * N_align * sizeof(int));
    // Load matrix data into one-dimensional arrays
    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&m1[i * K_align], &matrix1[i][0], K_align * sizeof(int));
    }

    for (size_t i = 0; i < K; ++i) {
        std::memcpy(&m2[i * N_align], &matrix2[i][0], N_align * sizeof(int));
    }

    // Tiled matrix multiplication
    const size_t blockSize = 64; // You can experiment with this value to match your cache size.
    
    __m256i temp;
    __m256i m2_vals;
    __m256i res_vals;
    __m256i prod;
    int tmp;
    for (size_t ii = 0; ii < M; ii += blockSize) {
        for (size_t kk = 0; kk < K; kk += blockSize) {
            for (size_t jj = 0; jj < N_align; jj += blockSize) {
                // Process the current block
                for (size_t i = ii; i < std::min(ii + blockSize, M); ++i) {
                    for (size_t k = kk; k < std::min(kk + blockSize, K); ++k) {
                        temp = _mm256_set1_epi32(m1[i * K_align + k]);  // Broadcast single element of m1
                        size_t j;
                        // SIMD loop to handle aligned elements
                        for ( j = jj; j < std::min(jj + blockSize, N_align); j += 8) {
                            m2_vals = _mm256_load_si256((__m256i*)&m2[k * N_align + j]);
                            res_vals = _mm256_load_si256((__m256i*)&res[i * N_align + j]);

                            prod = _mm256_mullo_epi32(temp, m2_vals);
                            res_vals = _mm256_add_epi32(res_vals, prod);

                            _mm256_store_si256((__m256i*)&res[i * N_align + j], res_vals);
                        }
                    }
                }
            }
        }
    }
    // for (size_t i = 0; i < M; ++i) {
    //     for (size_t k = 0; k < K; ++k) {
    //         __m256i temp = _mm256_set1_epi32(m1[i * K + k]);
    //         for (size_t j = 0; j < N; j += 8) {
    //             __m256i m2_vals = _mm256_loadu_si256((__m256i*)&m2[k * N + j]);
    //             __m256i res_vals = _mm256_loadu_si256((__m256i*)&res[i * N + j]);
    //             __m256i prod = _mm256_mullo_epi32(temp, m2_vals);
    //             res_vals = _mm256_add_epi32(res_vals, prod);
    //             _mm256_storeu_si256((__m256i*)&res[i * N + j], res_vals);
    //         }
    //     }
    // }

    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&result[i][0], &res[i * N_align], N_align * sizeof(int));
    }  
    _mm_free(m1);
    _mm_free(m2);
    _mm_free(res);
    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}