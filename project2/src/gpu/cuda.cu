//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Matrix Multiplication with CUDA, for bonus
//
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h> // CUDA runtime header

#include "../matrix.hpp" 

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(int* A, int* B, int* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

std::vector<int> matrix_multiply_cuda(const Matrix& matrix1, const Matrix& matrix2,const std::vector<int>& m1, const std::vector<int>& m2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        printf("Matrix dimensions are not compatible for multiplication.");
        exit(0);
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    std::vector<int> res(M * N);
    // Allocate memory for matrices on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));
    cudaError_t err;
    // Copy matrices from host to device
    cudaMemcpy(d_A, m1.data(), M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, m2.data(), K * N * sizeof(int), cudaMemcpyHostToDevice);
    // Define grid and block sizes
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    // Launch kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Check for any kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(0);
        //return Matrix(0, 0); // Return an empty matrix in case of error
    }

    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    auto elapsed_time = gpuDuration;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time
              << " milliseconds\n";
    // Copy result back from device to host
    err= cudaMemcpy(res.data(), d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
    std::cerr << "Failed to copy matrix A to device, error: " << cudaGetErrorString(err) << std::endl;
        exit(0);
        //return -1;
    }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return res;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
            return 0;
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];
    const std::string result_path = argv[3];

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

    res = matrix_multiply_cuda(matrix1, matrix2,m1,m2);
    Matrix result(M,N); 
    for(size_t i=0;i<M;i++){
        for(size_t j=0;j<N;j++){
            result[i][j]=res[i * N + j];
        }
    }
    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    return 0;
}