//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Reordering Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstring>
#include <immintrin.h>
Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);

    // Initialize the result matrix
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result[i][j] = 0;
        }
    }

    // Allocate and initialize one-dimensional arrays for m1 and m2
    int* m1 = (int*)_mm_malloc(M * K * sizeof(int), 32); 
    int* m2 = (int*)_mm_malloc(K * N * sizeof(int), 32);
    int* res = (int*)_mm_malloc(M * N * sizeof(int), 32);
    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&m1[i * K], &matrix1[i][0], K * sizeof(int));
    }

    // #pragma omp parallel for
    for (size_t i = 0; i < K; ++i) {
        std::memcpy(&m2[i * N], &matrix2[i][0], N * sizeof(int));
    }
    memset(res, 0, M * N * sizeof(int));
    // Tiled matrix multiplication
    const size_t blockSize = 64; // You can experiment with this value to match your cache size.
    int temp;
    for (size_t ii = 0; ii < M; ii += blockSize) {
        for (size_t kk = 0; kk < K; kk += blockSize) {
            for (size_t jj = 0; jj < N; jj += blockSize) {
                for (size_t i = ii; i < std::min(ii + blockSize, M); ++i) {
                    for (size_t k = kk; k < std::min(kk + blockSize, K); ++k) {
                        temp = m1[i * K + k];
                        for (size_t j = jj; j < std::min(jj + blockSize, N); ++j) {
                            res[i * N + j] += temp * m2[k * N + j];
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&result[i][0], &res[i * N], N * sizeof(int));
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
            
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(matrix1, matrix2);

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