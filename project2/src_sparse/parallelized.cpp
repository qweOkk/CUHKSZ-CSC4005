#include <omp.h>
#include <mpi.h>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include "sparse_matrix.hpp"

#define MASTER 0


/*****************************/
/**    Your code is here    **/
/*****************************/

// compute another column and save it as a dense vector
size_t scatter(const SparseMatrix& A, SparseMatrix& C, int iCol, int beta, int *w, int *x, int mark, size_t nnz){

    for (auto pos = A.start_[iCol]; pos < A.start_[iCol + 1]; ++pos){
        int iRow = A.idx_[pos];
        if (w[iRow] < mark) {
            w[iRow] = mark;
            C.idx_[nnz++] = iRow;
            x[iRow] = beta * A.val_[pos];
        }
        else x[iRow] += beta * A.val_[pos];
    }
    return nnz;
}
#include <omp.h>

SparseMatrix matrix_multiply(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.n_col_ != B.n_row_) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t n_row = A.n_row_, n_col = B.n_col_, nnz_a = A.val_.size(), nnz_b = B.val_.size();
    std::vector<int> w(n_row, 0);  // Must be thread-local
    std::vector<int> x(n_row);

    SparseMatrix C;
    C.n_row_ = n_row;
    C.n_col_ = n_col;
    C.start_.resize(n_col + 1);
    C.ensure_nnz_room(nnz_a + nnz_b, nnz_a + nnz_b);

    size_t total_nnz = 0;

    #pragma omp parallel for reduction(+:total_nnz) schedule(dynamic)
    for (int iCol = 0; iCol < n_col; ++iCol) {
        size_t nnz = 0;  // Thread-local nnz
        std::vector<int> w_local(n_row, 0);  // Each thread has its own w and x
        std::vector<int> x_local(n_row);

        C.ensure_nnz_room(total_nnz + n_row, 2 * C.val_.size() + n_row);
        C.start_[iCol] = total_nnz;

        // Compute the new column and scatter it into a dense vector
        for (auto pos = B.start_[iCol]; pos < B.start_[iCol + 1]; ++pos) {
            nnz = scatter(A, C, B.idx_[pos], B.val_[pos], w_local.data(), x_local.data(), iCol + 1, nnz);
        }

        // Gather dense vector components
        for (auto pos = C.start_[iCol]; pos < nnz; ++pos) {
            C.val_[pos] = x_local[C.idx_[pos]];
        }

        total_nnz += nnz;
    }

    C.start_[n_col] = total_nnz;
    C.idx_.resize(total_nnz);
    C.val_.resize(total_nnz);
    return C;
}



int main(int argc, char** argv) {
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

    SparseMatrix matrix1, matrix2;

    matrix1.loadFromFile(matrix1_path);
    matrix2.loadFromFile(matrix2_path);
    size_t M=matrix1.n_row_;
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
    auto start_time = std::chrono::high_resolution_clock::now();
    if (taskid == MASTER) {
        SparseMatrix result = matrix_multiply(matrix1, matrix2);

        // Your Code Here for Synchronization!

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                      start_time);

        result.save2File(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        SparseMatrix result = matrix_multiply(matrix1, matrix2);

        // Your Code Here for Synchronization!
    }

    MPI_Finalize();
    return 0;
}
