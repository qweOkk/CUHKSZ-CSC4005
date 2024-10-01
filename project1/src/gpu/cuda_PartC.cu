//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils.hpp"

/**
 * Demo kernel device function to clamp pixel value
 * 
 * You may mimic this to implement your own kernel device functions
 */
__device__ unsigned char d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}
// CUDA Kernel
// __global__ void bilateral_filter_kernel(const ColorValue* input, ColorValue* output, int width, int height, float sigma_d, float sigma_r) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     if (col >= width || row >= height) return;
//     // Spatial Weights
//     float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
//     float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));
//     float filtered_value=0.0;
//     if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
//         ColorValue center_value = input[row * width + col];
        
//         // Initialize weights and weighted values
//         float sum_weights = 0.0f;
//         float weighted_value_sum = 0.0f;
//         // Define the neighborhood
//         ColorValue value_11 = input[(row - 1) * width + (col - 1)];
//         ColorValue value_12 = input[(row - 1) * width + col];
//         ColorValue value_13 = input[(row - 1) * width + (col + 1)];
//         ColorValue value_21 = input[row * width + (col - 1)];
//         ColorValue value_22 = center_value;
//         ColorValue value_23 = input[row * width + (col + 1)];
//         ColorValue value_31 = input[(row + 1) * width + (col - 1)];
//         ColorValue value_32 = input[(row + 1) * width + col];
//         ColorValue value_33 = input[(row + 1) * width + (col + 1)];

//         // Calculate weights
//         float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_22 = 1.0;
//         float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) / (-2 * powf(SIGMA_R, 2)));
//         float w_33 = w_spatial_corner * expf(powf(center_value - value_33, 2) / (-2 * powf(SIGMA_R, 2))); 
//         // Sum of weights
//         sum_weights = w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;

//         // Calculate weighted value sum
//         weighted_value_sum = w_11 * value_11 + w_12 * value_12 + w_13 * value_13 +
//                             w_21 * value_21 + w_22 * value_22 + w_23 * value_23 +
//                             w_31 * value_31 + w_32 * value_32 + w_33 * value_33; 
//         // Calculate filtered value
//         filtered_value = weighted_value_sum / sum_weights;
//     }



//     output[row * width + col] = d_clamp_pixel_value(filtered_value);

// }
__global__ void bilateral_filter_kernel(const ColorValue* input, ColorValue* output, int width, int height, float sigma_d, float sigma_r) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        ColorValue center_value = input[row * width + col];
        float sum_weights = 0.0f;
        float filtered_value = 0.0f;

        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                ColorValue neighbor_value = input[(row + j) * width + (col + i)];
                float spatial_weight = expf(-0.5f * (i * i + j * j) / (sigma_d * sigma_d));
                float intensity_weight = expf(-(center_value - neighbor_value) * (center_value - neighbor_value) / (2.0f * sigma_r * sigma_r));
                float weight = spatial_weight * intensity_weight;
                
                sum_weights += weight;
                filtered_value += weight * neighbor_value;
            }
        }

        output[row * width + col] = d_clamp_pixel_value(filtered_value / sum_weights);

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
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // Apply the filter to the image

    // Apply the filter to the image
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    ColorValue* d_input_r,*d_input_g,*d_input_b;
    ColorValue* d_output_r,*d_output_g,*d_output_b;
    size_t size = width * height * sizeof(ColorValue);
    cudaMalloc(&d_input_r, size);
    cudaMalloc(&d_input_g, size);
    cudaMalloc(&d_input_b, size);
    cudaMalloc(&d_output_r, size);
    cudaMalloc(&d_output_g, size);
    cudaMalloc(&d_output_b, size);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaMemcpy(d_input_r, input_jpeg.r_values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_g, input_jpeg.g_values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, input_jpeg.b_values, size, cudaMemcpyHostToDevice);
    //auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: CUDA PartC
     */
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    bilateral_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_input_r, d_output_r, width, height, SIGMA_D, SIGMA_R);
    bilateral_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_input_b, d_output_b, width, height, SIGMA_D, SIGMA_R);
    bilateral_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_input_g, d_output_g, width, height, SIGMA_D, SIGMA_R);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    cudaMemcpy(output_jpeg.r_values, d_output_r, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_jpeg.b_values, d_output_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_jpeg.g_values, d_output_g, size, cudaMemcpyDeviceToHost);
    //auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = gpuDuration;
    //    end_time - start_time);
    // Save output JPEG image
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
    cudaFree(d_input_r);
    cudaFree(d_input_g);
    cudaFree(d_input_b);
    cudaFree(d_output_r);
    cudaFree(d_output_g);
    cudaFree(d_output_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time
              << " milliseconds\n";
    return 0;
}
