//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#include <cstring>
#define MASTER 0


Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, size_t start_row, size_t end_row) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    size_t M_align=((M-1)/8)*8+8;
    size_t N_align=((N-1)/8)*8+8;
    size_t K_align=((K-1)/8)*8+8;
    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division
        // Allocate and initialize one-dimensional arrays for m1 and m2
    int* m1 = (int*)_mm_malloc(M * K_align * sizeof(int), 32); 
    int* m2 = (int*)_mm_malloc(K * N_align * sizeof(int), 32);
    int* res = (int*)_mm_malloc(M * N_align * sizeof(int), 32);
    size_t row_size = N_align * sizeof(int);
    for (size_t i = start_row; i < end_row; ++i) {
        memset(&res[i * N_align], 0, row_size);  // Set each row to zero
    }
    // Load matrix data into one-dimensional arrays
    for (size_t i = start_row; i < end_row; ++i) {
        std::memcpy(&m1[i * K_align], &matrix1[i][0], K_align * sizeof(int));
    }

    for (size_t i = 0; i < K; ++i) {
        std::memcpy(&m2[i * N_align], &matrix2[i][0], N_align * sizeof(int));
    }

    // Each process only works on its own part of the matrix
    #pragma omp parallel
    {
        #pragma omp for collapse(1)
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t k = 0; k < K; ++k) {
                __m256i temp = _mm256_set1_epi32(m1[i * K_align + k]);  // Broadcast single element of m1
                size_t j;
                #pragma omp simd
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
    }

    // Copy the result back to the Matrix object
    for (size_t i = start_row; i < end_row; ++i) {
        std::memcpy(&result[i][0], &res[i * N_align], N_align * sizeof(int));
    }

    _mm_free(m1);
    _mm_free(m2);
    _mm_free(res);
    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    size_t M_align=((M-1)/8)*8+8;
    size_t N_align=((N-1)/8)*8+8;
    size_t K_align=((K-1)/8)*8+8;
    size_t block_size = M / numtasks; 
    // Calculate base rows and extra rows
    size_t base_rows = M / numtasks;
    size_t extra_rows = M % numtasks;

    // Calculate start_row and end_row for each process
    size_t start_row = taskid < extra_rows ? taskid * (base_rows + 1) : extra_rows * (base_rows + 1) + (taskid - extra_rows) * base_rows;
    size_t end_row = start_row + (taskid < extra_rows ? base_rows + 1 : base_rows);

    // Ensure end_row does not exceed M
    if (end_row > M) {
        end_row = M;
    }

    // Print debugging information
    //std::cout << "Process " << taskid << ": start_row = " << start_row << ", end_row = " << end_row << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        Matrix partial_result = matrix_multiply_mpi(matrix1, matrix2, start_row, end_row);
        Matrix result(M,N);
        // Your Code Here for Synchronization!
        int* res = (int*)_mm_malloc(M * N_align * sizeof(int), 32);
        for (size_t i = start_row; i < end_row; ++i) {
            std::memcpy(&result[i][0], &partial_result[i][0], N * sizeof(int));
        }

        // Receive and aggregate results from other processes
        for (int p = 1; p < numtasks; ++p) {
            size_t other_start_row = p < extra_rows ? p * (base_rows + 1) : extra_rows * (base_rows + 1) + (p - extra_rows) * base_rows;
            size_t other_end_row = other_start_row + (p < extra_rows ? base_rows + 1 : base_rows);
            if (other_end_row > M) {
                other_end_row = M;
            }
            int length = (other_end_row - other_start_row) * N_align;
            //std::cout << "Receiving from process " << p << " with length " << length << " from row " << other_start_row << " to row " << other_end_row << std::endl;
            MPI_Recv(&res[other_start_row*N_align], length, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t i = other_start_row; i < other_end_row; ++i) {
                std::memcpy(&result[i][0], &res[i * N_align], N_align * sizeof(int));
            } 
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        Matrix partial_result = matrix_multiply_mpi(matrix1, matrix2, start_row, end_row);
        // Send the partial result back to the master process
        int length = (end_row - start_row) * N_align;
        //std::cout << "Sending from process " << taskid << " block_size "<<block_size<<" with length " << length << " from row " << start_row << " to row " << end_row << std::endl;
        int* res = (int*)_mm_malloc(M * N_align * sizeof(int), 32);
        for (size_t i = start_row; i < end_row; ++i) {
            std::memcpy(&res[i * N_align], &partial_result[i][0], N_align * sizeof(int));
        }   
        MPI_Send(&res[start_row*N_align], length, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        // Your Code Here for Synchronization!
    }

    MPI_Finalize();
    return 0;
}