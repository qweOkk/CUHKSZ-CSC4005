#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <atomic>
#include "../utils.hpp"

// Global task index to track the current row being processed
std::atomic<int> global_row_index(0);

struct ThreadData
{
    JpegSOA input_jpeg;
    JpegSOA output_jpeg;
    int width;
    int height;
    int num_channels;
    int rows_per_task; // Number of rows each task processes
};

void* bilateral_filter_thread_function(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    int row;

    // Dynamic task allocation
    while ((row = global_row_index.fetch_add(data->rows_per_task)) < data->height)
    {
        int end_row = std::min(row + data->rows_per_task, data->height); // Ensure not to exceed image height

        for (int r = row; r < end_row; r++)
        {
            for (int col = 0; col < data->width; col++)
            {
                for (int channel = 0; channel < data->num_channels; ++channel)
                {
                    int index = r * data->width + col;
                    ColorValue filtered_value = bilateral_filter(
                        data->input_jpeg.get_channel(channel), r, col, data->width);
                    data->output_jpeg.set_value(channel, index, filtered_value);
                }
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

    // Allocate memory for the output
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};

    int NUM_THREADS = std::stoi(argv[3]); // Convert the input to integer
    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];

    int rowsPerTask = 10; // Each task processes 10 rows

    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadData[i] = {input_jpeg,
                         output_jpeg,
                         input_jpeg.width,
                         input_jpeg.height,
                         input_jpeg.num_channels,
                         rowsPerTask};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, bilateral_filter_thread_function,
                       &threadData[i]);
    }

    // Join threads
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

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
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
