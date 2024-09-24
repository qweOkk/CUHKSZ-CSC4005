//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

void set_filtered_image(JpegSOA& input_jpeg,
                        std::vector<ColorValue>& output_r_values,
                        std::vector<ColorValue>& output_g_values,
                        std::vector<ColorValue>& output_b_values,
                        int width, int num_channels, int start_line,
                        int end_line)
{
    for (int row = start_line; row < end_line; row++)
    {
        for (int col = 1; col < width - 1; col++)
        {
            for (int channel = 0; channel < num_channels; ++channel)
            {
                int index = row * width + col;
                ColorValue filtered_value = bilateral_filter(
                    input_jpeg.get_channel(channel), row, col, width);

                if (channel == 0) output_r_values[index] = filtered_value;
                if (channel == 1) output_g_values[index] = filtered_value;
                if (channel == 2) output_b_values[index] = filtered_value;
            }
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

    std::cout << "Input file from: " << input_filepath << "\n";
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

    int total_line_num = height - 2;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    // Calculate the cut-off points for each process
    std::vector<int> cuts(numtasks + 1, 1);
    for (int i = 0; i < numtasks; i++)
    {
        cuts[i + 1] = cuts[i] + line_per_task + (i < left_line_num ? 1 : 0);
    }

    if (taskid == MASTER) 
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Master process handles its own slice
        set_filtered_image(input_jpeg, output_r_values, output_g_values, output_b_values, width, num_channels, cuts[taskid], cuts[taskid + 1]);

        // Allocate space to receive data from other processes
        for (int i = 1; i < numtasks; i++) 
        {
            int start_index = cuts[i] * width;
            int length = width * (cuts[i + 1] - cuts[i]);

            // Receive filtered data for each channel from other processes
            MPI_Recv(output_r_values.data() + start_index, length, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(output_g_values.data() + start_index, length, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(output_b_values.data() + start_index, length, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";

        // Use output_jpeg object in export_jpeg
        if (export_jpeg(output_jpeg, output_filepath)) 
        {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    } 
    else 
    {
        // Worker processes handle their own slice
        int offset = cuts[taskid] * width;
        int length = width * (cuts[taskid + 1] - cuts[taskid]);

        set_filtered_image(input_jpeg, output_r_values, output_g_values, output_b_values, width, num_channels, cuts[taskid], cuts[taskid + 1]);

        // Send filtered data back to master
        MPI_Send(output_r_values.data() + offset, length, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(output_g_values.data() + offset, length, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(output_b_values.data() + offset, length, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
