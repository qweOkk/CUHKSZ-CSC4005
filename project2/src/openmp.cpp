//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// OpenMp + SIMD + Reordering Matrix Multiplication
//scan

#include <immintrin.h>
#include <omp.h> 
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstring>
Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);
    size_t M_align=((M-1)/8)*8+8;
    size_t N_align=((N-1)/8)*8+8;
    size_t K_align=((K-1)/8)*8+8;
    // Initialize the result matrix
    // //  #pragma omp parallel for collapse(2)
    // for (size_t i = 0; i < M; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         result[i][j] = 0;
    //     }
    // }
    // return result;
    // Allocate and initialize one-dimensional arrays for m1 and m2
    int* m1 = (int*)_mm_malloc(M * K_align * sizeof(int), 32); 
    int* m2 = (int*)_mm_malloc(K * N_align * sizeof(int), 32);
    int* res = (int*)_mm_malloc(M * N_align * sizeof(int), 32);

    // Load matrix data into one-dimensional arrays
    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&m1[i * K_align], &matrix1[i][0], K_align * sizeof(int));
    }

    for (size_t i = 0; i < K; ++i) {
        std::memcpy(&m2[i * N_align], &matrix2[i][0], N_align * sizeof(int));
    }

//  
    // Initialize res to zero
    memset(res, 0, M * N_align * sizeof(int));

    // #pragma omp parallel
    // {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                __m256i temp = _mm256_set1_epi32(m1[i * K_align + k]);  // Broadcast single element of m1
                size_t j;
                // SIMD loop to handle aligned elements
                for (j = 0; j < N_align; j += 8) {
                    __m256i m2_vals = _mm256_load_si256((__m256i*)&m2[k * N_align + j]);
                    __m256i res_vals = _mm256_load_si256((__m256i*)&res[i * N_align + j]);

                    __m256i prod = _mm256_mullo_epi32(temp, m2_vals);
                    res_vals = _mm256_add_epi32(res_vals, prod);

                    _mm256_store_si256((__m256i*)&res[i * N_align + j], res_vals);
                }

                // int tmp = m1[i * K + k];
                // for (; j < N; ++j) {
                //     res[i * N + j] += tmp * m2[k * N + j];
                // }
            }
        }
    // }

    // auto start = std::chrono::high_resolution_clock::now();
//  #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&result[i][0], &res[i * N_align], N_align * sizeof(int));
    }

    _mm_free(m1);
    _mm_free(m2);
    _mm_free(res);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "复制时间: " << duration << " 微秒" << std::endl;
    return result;
}
int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_openmp(matrix1, matrix2);

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