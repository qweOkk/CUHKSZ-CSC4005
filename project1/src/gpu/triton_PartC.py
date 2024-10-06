import triton
import triton.language as tl
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Triton kernel for Bilateral Filter
@triton.jit
def bilateral_filter_kernel(input_ptr, output_ptr, width, height, num_channels, sigma_d, sigma_r, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (idx < (width * height)) & (idx % width > 0) & (idx % width < width - 1) & (idx // width > 0) & (idx // width < height - 1)

    for c in range(num_channels):  # Apply the filter for each color channel
        center_values = tl.load(input_ptr + (idx[:, None] * num_channels + c), mask=mask[:, None]).to(tl.float32)
        sum_weights = tl.zeros_like(center_values)
        filtered_values = tl.zeros_like(center_values)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                neighbor_indices = idx + dy * width + dx
                neighbor_values = tl.load(input_ptr + (neighbor_indices[:, None] * num_channels + c), mask=mask[:, None]).to(tl.float32)
                spatial_weights = tl.exp(-0.5 * (dx * dx + dy * dy) / (sigma_d * sigma_d))
                intensity_weights = tl.exp(-(center_values - neighbor_values) * (center_values - neighbor_values) / (2.0 * sigma_r * sigma_r))
                weights = spatial_weights * intensity_weights
                sum_weights += weights
                filtered_values += weights * neighbor_values

        filtered_values /= sum_weights
        filtered_values = tl.where(filtered_values > 255.0, 255.0, filtered_values)
        filtered_values = tl.where(filtered_values < 0.0, 0.0, filtered_values)
        filtered_values = filtered_values.to(tl.uint8)
        tl.store(output_ptr + (idx[:, None] * num_channels + c), filtered_values, mask=mask[:, None])

def read_from_jpeg(filepath):
    image = Image.open(filepath)
    image = image.convert('RGB')
    image_array = torch.tensor(image.getdata(), dtype=torch.uint8).reshape(image.size[1], image.size[0], 3)
    return image_array, image.width, image.height, 3

def export_jpeg(image_tensor, filepath):
    image_array = image_tensor.byte().cpu().contiguous()
    image = Image.fromarray(image_array.numpy(), mode='RGB')
    image.save(filepath)

def main(input_filepath, output_filepath):
    # Read from input JPEG
    print(f"Input file from: {input_filepath}")
    input_image, width, height, num_channels = read_from_jpeg(input_filepath)
    for _ in range(20):
        # Allocate memory for output image and buffer
        output_image = torch.empty_like(input_image)
        buffer = input_image.flatten()

        # Copy data to GPU
        buffer_ptr = buffer.cuda()
        output_ptr = output_image.flatten().cuda()

        sigma_d = 1.7
        sigma_r = 50.0

        # Launch Triton kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        grid = ((width * height + 1024 - 1024) // 1024,)
        bilateral_filter_kernel[grid](buffer_ptr, output_ptr, width, height, num_channels, sigma_d, sigma_r, BLOCK_SIZE=1024)
        end_event.record()
        torch.cuda.synchronize()

        # Copy result back to host
        output_image = output_ptr.cpu().reshape((height, width, num_channels))
        # Free GPU memory
        del buffer_ptr
        del output_ptr
        torch.cuda.empty_cache()

    # Write output image to output JPEG
    print(f"Output file to: {output_filepath}")
    export_jpeg(output_image, output_filepath)

    # Print execution time
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Transformation Complete!")
    print(f"Execution Time: {elapsed_time_ms:.2f} milliseconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])