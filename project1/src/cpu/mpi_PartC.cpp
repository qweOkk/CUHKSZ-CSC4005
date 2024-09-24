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
void set_filtered_image(JpegSOA output_jpeg, JpegSOA input_jpeg,
                        int width, int num_chanels, int start_line,
                        int end_line, int offset)
{
    for (int row = start_line; row < end_line; row++)
    {
        for (int col = 1; col < width - 1; col++)
        {
            for (int channel = 0; channel < input_jpeg.num_channels; ++channel){
                int index = (row * width + col);
                ColorValue filtered_value = bilateral_filter(
                input_jpeg.get_channel(channel), row, col, width);
                output_jpeg.set_value(channel, index, filtered_value);
            }
            
            // float r_sum =
            //     linear_filter(image, filter, r_id, width, num_chanels);

            // float g_sum =
            //     linear_filter(image, filter, g_id, width, num_chanels);

            // float b_sum =
            //     linear_filter(image, filter, b_id, width, num_chanels);

            // filtered_image[r_id - offset] = clamp_pixel_value(r_sum);
            // filtered_image[g_id - offset] = clamp_pixel_value(g_sum);
            // filtered_image[b_id - offset] = clamp_pixel_value(b_sum);
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
    // Divide the task
    // For example, there are 11 lines and 3 tasks,
    // we try to divide to 4 4 3 instead of 3 3 5
    std::cout << "Input file from: " << input_filepath << "\n";
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
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
    int total_line_num = input_jpeg.height - 2;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 1);
    int divided_left_line_num = 0;
    for (int i = 0; i < numtasks; i++)
    {
        if (divided_left_line_num < left_line_num)
        {
            cuts[i + 1] = cuts[i] + line_per_task + 1;
            divided_left_line_num++;
        }
        else
            cuts[i + 1] = cuts[i] + line_per_task;
    }
    if (taskid == MASTER) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 主进程处理自己负责的分片
        set_filtered_image(output_jpeg, input_jpeg, width, num_channels, cuts[taskid], cuts[taskid + 1], 0);

        // 接收其他进程处理的数据
        for (int i = 1; i < numtasks; i++) {
            int start_index = cuts[i] * width;
            int length = width * (cuts[i + 1] - cuts[i]);
            
            // 接收每个通道的数据
            MPI_Recv(output_r_values + start_index, length, MPI_UNSIGNED_CHAR , i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(output_g_values + start_index, length, MPI_UNSIGNED_CHAR , i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(output_b_values + start_index, length, MPI_UNSIGNED_CHAR , i, TAG_GATHER, MPI_COMM_WORLD, &status);
            
        }

        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        if (export_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }

        delete[] output_r_values;
        delete[] output_g_values;
        delete[] output_b_values;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } else {
        // 子进程处理自己的分片
        int length = width * (cuts[taskid + 1] - cuts[taskid]);
        int offset = cuts[taskid] * width;

        set_filtered_image(output_jpeg, input_jpeg, width, num_channels, cuts[taskid], cuts[taskid + 1], offset);

        // 发送处理后的数据回给主进程
        MPI_Send(output_r_values + offset, length, MPI_UNSIGNED_CHAR , MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(output_g_values + offset, length, MPI_UNSIGNED_CHAR , MASTER, TAG_GATHER, MPI_COMM_WORLD);
        MPI_Send(output_b_values + offset, length, MPI_UNSIGNED_CHAR , MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
