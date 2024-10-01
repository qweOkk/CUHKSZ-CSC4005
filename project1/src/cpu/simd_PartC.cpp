#include <immintrin.h>
#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include "../utils.hpp"

#include <immintrin.h>
#include <cmath>

#include <immintrin.h>
#include <cmath>

// Function to reduce the range of x to [-ln(2)/2, ln(2)/2]
__m256 reduce_range(__m256 x) {
    const float log2e = 1.4426950408889634f; // log2(e)
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(log2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.6931471805599453f))); // 0.6931471805599453 is ln(2)
    return r;
}

__m256 exp_ps(__m256 x) {
    // Reduce the range of x
    x = reduce_range(x);

    // Constants for the higher-order Padé approximant
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
    const __m256 c6 = _mm256_set1_ps(1.0f / 720.0f);
    const __m256 c7 = _mm256_set1_ps(1.0f / 5040.0f);
    const __m256 d1 = _mm256_set1_ps(1.0f);
    const __m256 d2 = _mm256_set1_ps(-0.5f);
    const __m256 d3 = _mm256_set1_ps(1.0f / 12.0f);
    const __m256 d4 = _mm256_set1_ps(-1.0f / 120.0f);
    const __m256 d5 = _mm256_set1_ps(1.0f / 2520.0f);

    // Calculate powers of x
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x4 = _mm256_mul_ps(x3, x);
    __m256 x5 = _mm256_mul_ps(x4, x);
    __m256 x6 = _mm256_mul_ps(x5, x);
    __m256 x7 = _mm256_mul_ps(x6, x);

    // Numerator and denominator of the higher-order Padé approximant
    __m256 numerator = c0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5 + c6 * x6 + c7 * x7;
    __m256 denominator = d1 + d2 * x2 + d3 * x4 + d4 * x6 + d5 * x7;

    // Compute the Padé approximant
    __m256 result = _mm256_div_ps(numerator, denominator);

    // Reconstruct the original value using 2^n * exp(r)
    const float log2e = 1.4426950408889634f; // log2(e)
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(log2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    result = _mm256_mul_ps(result, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtps_epi32(n), 23)));

    return result;
}

void load_values(const unsigned char* values, int index, __m256* center_value) {
     __m128i uchar_values = _mm_loadl_epi64((__m128i*)&values[index]); 

    __m256i int_values = _mm256_cvtepu8_epi32(uchar_values);

    *center_value = _mm256_cvtepi32_ps(int_values); 
}
ColorValue bilateral_filter_avx2(const ColorValue* values, int row, int col, int width, float* filtered_values) {
    const float inv_sigma_d2 = -0.5 / (SIGMA_D * SIGMA_D);
    const float inv_sigma_r2 = -0.5 / (SIGMA_R * SIGMA_R);
    __m256 center_value;
    int index = row * width + col;
    load_values(values,index,&center_value);
    // Load center pixel value
    // __m256 center_value = _mm256_set1_ps(values[row * width + col]);

    // Spatial Weights
    __m256 w_spatial_border = _mm256_set1_ps(expf(inv_sigma_d2));
    __m256 w_spatial_corner = _mm256_set1_ps(expf(-1.0 / (SIGMA_D * SIGMA_D)));

    // Load neighbor pixel values

    __m256 value_11, value_12, value_13, value_21, value_22, value_23, value_31, value_32, value_33;
    load_values(values, (row - 1) * width + (col - 1), &value_11);
    load_values(values, (row - 1) * width + col, &value_12);
    load_values(values, (row - 1) * width + (col + 1), &value_13);
    load_values(values, row * width + (col - 1), &value_21);
    load_values(values, row * width + col, &value_22);
    load_values(values, row * width + (col + 1), &value_23);
    load_values(values, (row + 1) * width + (col - 1), &value_31);
    load_values(values, (row + 1) * width + col, &value_32);
    load_values(values, (row + 1) * width + (col + 1), &value_33);
    // Calculate intensity differences
    __m256 diff_11 = _mm256_sub_ps(center_value, value_11);
    __m256 diff_12 = _mm256_sub_ps(center_value, value_12);
    __m256 diff_13 = _mm256_sub_ps(center_value, value_13);
    __m256 diff_21 = _mm256_sub_ps(center_value, value_21);
    __m256 diff_23 = _mm256_sub_ps(center_value, value_23);
    __m256 diff_31 = _mm256_sub_ps(center_value, value_31);
    __m256 diff_32 = _mm256_sub_ps(center_value, value_32);
    __m256 diff_33 = _mm256_sub_ps(center_value, value_33);

    // Calculate intensity weights
    __m256 w_11 = _mm256_mul_ps(w_spatial_corner, exp_ps(_mm256_mul_ps(diff_11, diff_11) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_12 = _mm256_mul_ps(w_spatial_border, exp_ps(_mm256_mul_ps(diff_12, diff_12) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_13 = _mm256_mul_ps(w_spatial_corner, exp_ps(_mm256_mul_ps(diff_13, diff_13) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_21 = _mm256_mul_ps(w_spatial_border, exp_ps(_mm256_mul_ps(diff_21, diff_21) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_22 = _mm256_set1_ps(1.0f);
    __m256 w_23 = _mm256_mul_ps(w_spatial_border, exp_ps(_mm256_mul_ps(diff_23, diff_23) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_31 = _mm256_mul_ps(w_spatial_corner, exp_ps(_mm256_mul_ps(diff_31, diff_31) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_32 = _mm256_mul_ps(w_spatial_border, exp_ps(_mm256_mul_ps(diff_32, diff_32) * _mm256_set1_ps(inv_sigma_r2)));
    __m256 w_33 = _mm256_mul_ps(w_spatial_corner, exp_ps(_mm256_mul_ps(diff_33, diff_33) * _mm256_set1_ps(inv_sigma_r2)));

    // Sum of weights
    __m256 sum_weights = _mm256_add_ps(w_11, w_12);
    sum_weights = _mm256_add_ps(sum_weights, w_13);
    sum_weights = _mm256_add_ps(sum_weights, w_21);
    sum_weights = _mm256_add_ps(sum_weights, w_22);
    sum_weights = _mm256_add_ps(sum_weights, w_23);
    sum_weights = _mm256_add_ps(sum_weights, w_31);
    sum_weights = _mm256_add_ps(sum_weights, w_32);
    sum_weights = _mm256_add_ps(sum_weights, w_33);

    // Calculate weighted sum of values
    __m256 weighted_sum = _mm256_add_ps(_mm256_mul_ps(w_11, value_11), _mm256_mul_ps(w_12, value_12));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_13, value_13));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_21, value_21));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_22, value_22));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_23, value_23));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_31, value_31));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_32, value_32));
    weighted_sum = _mm256_add_ps(weighted_sum, _mm256_mul_ps(w_33, value_33));

    // Create an array to store the 8 pixel results
    __m256 result = _mm256_div_ps(weighted_sum, sum_weights);
    _mm256_storeu_ps(filtered_values, result);

    // 对结果进行clamp操作
    for (int i = 0; i < 8; ++i) {
        filtered_values[i] = clamp_pixel_value(filtered_values[i]);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr) {
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
    JpegSOA output_jpeg{output_r_values, output_g_values, output_b_values, width, height, num_channels, input_jpeg.color_space};

    auto start_time = std::chrono::high_resolution_clock::now();

for (int channel = 0; channel < num_channels; ++channel) {
    for (int row = 1; row < height - 1; ++row) {
        for (int col = 1; col < width - 7; col += 8) {
            int index = row * width + col;
            
            float filtered_values[8];
            bilateral_filter_avx2(input_jpeg.get_channel(channel), row, col, width, filtered_values);

            for (int i = 0; i < 8; ++i) {
                output_jpeg.set_value(channel, index + i, filtered_values[i]);
            }
            
        }
        for (int col = width - (width % 8)-8; col < width-1; ++col) {
            int index = row * width + col;
            ColorValue filtered_value = bilateral_filter(input_jpeg.get_channel(channel), row, col, width);
        output_jpeg.set_value(channel, index, filtered_value);
    }
    }
}

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath)) {
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