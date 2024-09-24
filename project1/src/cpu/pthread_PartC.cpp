//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"
struct ThreadData
{
    JpegSOA input_jpeg;
    JpegSOA output_jpeg;
    int width;
    int height;
    int num_channels;
    int start_row;
    int end_row;
};

void* grayscale_filter_thread_function(void* arg)
{
    ThreadData* data = (ThreadData*)arg;

    for (int row = data->start_row; row < data->end_row; row++)
    {
        for (int col = 0; col < data->width; col++)
        {   
            for (int channel = 0; channel < data->num_channels; ++channel){
                int index = row * data->width + col;
                ColorValue filtered_value = bilateral_filter(
                    data->input_jpeg.get_channel(channel), row, col, data->width);
                data->output_jpeg.set_value(channel, index, filtered_value);
            }

            
        }
    }
    return nullptr;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
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
    int NUM_THREADS = std::stoi(argv[3]); // Convert the input to integer
    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];
    int rowsPerThread = input_jpeg.height / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadData[i] = {input_jpeg,
                         output_jpeg,
                         input_jpeg.width,
                         input_jpeg.height,
                         input_jpeg.num_channels,
                         i * rowsPerThread,
                         (i == NUM_THREADS - 1) ? input_jpeg.height
                                                : (i + 1) * rowsPerThread};
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: Pthread PartC
     */
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, grayscale_filter_thread_function,
                       &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
