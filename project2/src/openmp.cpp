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

std::vector<int> matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2,const std::vector<int>& m1, const std::vector<int>& m2) {
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    // Matrix result(M, N);
    
    // Initialize the result matrix
    //  #pragma omp parallel for collapse(2)
    // for (size_t i = 0; i < M; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         result[i][j] = 0;
    //     }
    // }

    // Allocate and initialize one-dimensional arrays for m1 and m2
    // std::vector<int> m1(M * K);
    // std::vector<int> m2(K * N);
    std::vector<int> res(M * N);
    // //  #pragma omp parallel for collapse(2)
    // for (size_t i = 0; i < M; ++i) {
    //     for (size_t j = 0; j < K; ++j) {
    //         m1[i * K + j] = matrix1[i][j];
    //     }
    // }
    // //  #pragma omp parallel for collapse(2)
    // for (size_t i = 0; i < K; ++i) {
    //     for (size_t j = 0; j < N; ++j) {
    //         m2[i * N + j] = matrix2[i][j];
    //     }
    // }

    // Tiled matrix multiplication
    const size_t blockSize = 64; // You can experiment with this value to match your cache size.
    
#pragma omp parallel for schedule(static) collapse(3) 
    for (size_t ii = 0; ii < M; ii += blockSize) {
        for (size_t kk = 0; kk < K; kk += blockSize) {
            for (size_t jj = 0; jj < N; jj += blockSize) {
                // Process the current block
                for (size_t i = ii; i < std::min(ii + blockSize, M); ++i) {
                    for (size_t k = kk; k < std::min(kk + blockSize, K); ++k) {
                    __m256i temp = _mm256_set1_epi32(m1[i * K + k]);  // Broadcast single element of m1
                    size_t j;
                    size_t aligned_end_j = (N / 8) * 8; 
                    // SIMD loop to handle aligned elements
                    for ( j = jj; j < std::min(jj + blockSize, aligned_end_j); j += 8) {
                        __m256i m2_vals = _mm256_loadu_si256((__m256i*)&m2[k * N + j]);
                        __m256i res_vals = _mm256_loadu_si256((__m256i*)&res[i * N + j]);

                        __m256i prod = _mm256_mullo_epi32(temp, m2_vals);
                        res_vals = _mm256_add_epi32(res_vals, prod);

                        _mm256_storeu_si256((__m256i*)&res[i * N + j], res_vals);
                    }

                        int tmp=m1[i * K + k];
                        for (; j <std::min(N,jj+blockSize); ++j) {
                            res[i * N + j] += tmp * m2[k * N + j];
                        }
                    }
                }
            }
        }
    }


//  #pragma omp parallel for collapse(2)


    return res;
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
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    std::vector<int> m1(M * K);
    std::vector<int> m2(K * N);
    std::vector<int> res(M * N);
    //  #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            m1[i * K + j] = matrix1[i][j];
        }
    }
    //  #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m2[i * N + j] = matrix2[i][j];
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    res= matrix_multiply_openmp(matrix1, matrix2,m1,m2);

    auto end_time = std::chrono::high_resolution_clock::now();
    Matrix result(M,N); 
    for(size_t i=0;i<M;i++){
        for(size_t j=0;j<N;j++){
            result[i][j]=res[i * N + j];
        }
    }
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}