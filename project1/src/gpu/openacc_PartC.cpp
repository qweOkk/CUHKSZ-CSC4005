//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

#pragma acc routine seq
ColorValue bilateral_filter_acc(const ColorValue* values, int row, int col, int width,int height)
{   
    if (row <= 0 || row >= height - 1 || col <= 0 || col >= width - 1) {
        return values[row * width + col]; 
    }

    int index = row * width + col;
    if (index >= 0 && index < width * height){
        ColorValue value_11 = values[(row - 1) * width + (col - 1)];
        ColorValue value_12 = values[(row - 1) * width + col];
        ColorValue value_13 = values[(row - 1) * width + (col + 1)];
        ColorValue value_21 = values[row * width + (col - 1)];
        ColorValue value_22 = values[row * width + col];
        ColorValue value_23 = values[row * width + (col + 1)];
        ColorValue value_31 = values[(row + 1) * width + (col - 1)];
        ColorValue value_32 = values[(row + 1) * width + col];
        ColorValue value_33 = values[(row + 1) * width + (col + 1)];
        // Spatial Weights
        float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
        float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));
        // Intensity Weights
        ColorValue center_value = value_22;
        float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_22 = 1.0;
        float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) / (-2 * powf(SIGMA_R, 2)));
        float w_33 = w_spatial_corner * expf(powf(center_value - value_33, 2) / (-2 * powf(SIGMA_R, 2)));
        float sum_weights = w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;
        // Calculate filtered value
        float filtered_value = (w_11 * value_11 + w_12 * value_12 + w_13 * value_13 + w_21 * value_21 +
                                w_22 * center_value + w_23 * value_23 + w_31 * value_31 +
                                w_32 * value_32 + w_33 * value_33) / sum_weights;
        return acc_clamp_pixel_value(filtered_value);
    }
    return 0;
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
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    // Apply the filter to the image using OpenACC
    
#pragma acc data copyin(input_jpeg) create(output_jpeg)
#pragma acc update device(input_jpeg) 
#pragma acc parallel present(input_jpeg,output_jpeg) num_gangs(1024)
    auto start_time = std::chrono::high_resolution_clock::now();
        #pragma acc loop independent
        for (int row = 1; row < height - 1; ++row)
        {
            #pragma acc loop independent
            for (int col = 1; col < width - 1; ++col)
            {   

                int index = row * width + col;
                if (index >= 0 && index < (width-1) * (height-1)){
                    ColorValue filtered_value_r = bilateral_filter_acc(
                        input_jpeg.r_values, row, col, width,height);
                    ColorValue filtered_value_g = bilateral_filter_acc(
                        input_jpeg.g_values, row, col, width,height);
                    ColorValue filtered_value_b = bilateral_filter_acc(
                        input_jpeg.b_values, row, col, width,height);
                    output_jpeg.r_values[index]=filtered_value_r;
                    output_jpeg.g_values[index]=filtered_value_g;
                    output_jpeg.b_values[index]=filtered_value_b;
                }
                
            }
        }



    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);
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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
