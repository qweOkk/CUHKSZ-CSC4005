#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header
#include <omp.h>
#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

// Optimized function to set the filtered image
void set_filtered_image_optimized(JpegSOA& input_jpeg,
                                  std::vector<ColorValue>& output_r_values,
                                  std::vector<ColorValue>& output_g_values,
                                  std::vector<ColorValue>& output_b_values,
                                  int width, int num_channels, int start_row,
                                  int block_height)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int row = start_row; row < start_row + block_height; row++) {
        for (int col = 1; col < width - 1; col++) {
            // Cache index calculation
            int index = row * width + col;

            // Cache input channel values into local variables to avoid redundant memory access
            ColorValue r_value = input_jpeg.r_values[index];
            ColorValue g_value = input_jpeg.g_values[index];
            ColorValue b_value = input_jpeg.b_values[index];

            // Process each channel
            output_r_values[index] = bilateral_filter(input_jpeg.r_values, row, col, width);
            output_g_values[index] = bilateral_filter(input_jpeg.g_values, row, col, width);
            output_b_values[index] = bilateral_filter(input_jpeg.b_values, row, col, width);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    
    // Read input JPEG File
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;

    // Use std::vector for output arrays to avoid manual memory management
    std::vector<ColorValue> output_r_values(width * height);
    std::vector<ColorValue> output_g_values(width * height);
    std::vector<ColorValue> output_b_values(width * height);

    JpegSOA output_jpeg{
        output_r_values.data(), output_g_values.data(), output_b_values.data(),
        width, height, num_channels, input_jpeg.color_space};

    // Start the MPI
    MPI_Init(&argc, &argv);
    int numtasks, taskid, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int block_height = height / numtasks; 
    int remainder = height % numtasks; 

    int extra_rows = taskid < remainder ? 1 : 0;
    int current_block_height = block_height + extra_rows;
    int start_row = taskid * block_height + std::min(taskid, remainder);

    if (taskid == MASTER) 
    {
        std::cout << "Input file from: " << input_filepath << "\n";

        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Master process handles its own slice
        set_filtered_image_optimized(input_jpeg, output_r_values, output_g_values, output_b_values, width, num_channels, start_row, current_block_height);

        // Prepare for receiving data from other processes
        std::vector<int> sendcounts(numtasks), displs(numtasks);
        for (int i = 0; i < numtasks; ++i) {
            int extra = i < remainder ? 1 : 0;
            sendcounts[i] = (block_height + extra) * width;
            displs[i] = i * block_height * width + std::min(i, remainder) * width;
        }

        // Non-blocking receives
        std::vector<MPI_Request> requests(3 * (numtasks - 1));
        for (int i = 1; i < numtasks; i++) {
            MPI_Irecv(output_r_values.data() + displs[i], sendcounts[i], MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &requests[3 * (i - 1)]);
            MPI_Irecv(output_g_values.data() + displs[i], sendcounts[i], MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &requests[3 * (i - 1) + 1]);
            MPI_Irecv(output_b_values.data() + displs[i], sendcounts[i], MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &requests[3 * (i - 1) + 2]);
        }

        // Wait for all receives to complete
        MPI_Waitall(3 * (numtasks - 1), requests.data(), MPI_STATUSES_IGNORE);

        auto end_time = std::chrono::high_resolution_clock::now();        
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

        // Use output_jpeg object in export_jpeg
        if (export_jpeg(output_jpeg, output_filepath)) 
        {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }

    } 
    else 
    {
        // Worker processes handle their own slice
        set_filtered_image_optimized(input_jpeg, output_r_values, output_g_values, output_b_values, width, num_channels, start_row, current_block_height);

        // Send filtered data back to master
        int sendcount = current_block_height * width;
        MPI_Request request;
        MPI_Isend(output_r_values.data() + start_row * width, sendcount, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD, &request);
        MPI_Isend(output_g_values.data() + start_row * width, sendcount, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD, &request);
        MPI_Isend(output_b_values.data() + start_row * width, sendcount, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD, &request);
    }

    MPI_Finalize();
    return 0;
}