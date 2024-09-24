//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray
// (Strcture-of-Array)
//

#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "../utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    int NUM_THREADS = std::stoi(argv[3]);
    omp_set_num_threads(NUM_THREADS);
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: OpenMP PartC
     */
// #pragma omp parallel for shared(input_jpeg, output_jpeg)

//     for (int row = 1; row < height - 1; ++row)
//         {
//             for (int col = 1; col < width - 1; ++col)
//             {
//                 for (int channel = 0; channel < num_channels; ++channel){
//                     int index = row * width + col;
//                     ColorValue filtered_value = bilateral_filter(
//                     input_jpeg.get_channel(channel), row, col, width);
//                     output_jpeg.set_value(channel, index, filtered_value);
//                 }
                
//             }
//         }
    #pragma omp parallel
    {

        int thread_id = omp_get_thread_num();
        int num_threads = NUM_THREADS;

        for (int row = thread_id + 1; row < height - 1; row += num_threads)
        {
            for (int col = 1; col < width - 1; ++col)
            {
                for (int channel = 0; channel < num_channels; ++channel)
                {
                    int index = row * width + col;
                    ColorValue filtered_value = bilateral_filter(
                        input_jpeg.get_channel(channel), row, col, width);
                    output_jpeg.set_value(channel, index, filtered_value);
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
