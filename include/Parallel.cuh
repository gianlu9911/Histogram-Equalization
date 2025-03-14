#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_BINS 256

#define TILE_WIDTH 32  // Define tile size



// CUDA kernel using tiling and shared memory for histogram computation
__global__ void computeHistogramTiled(const unsigned char* d_img, int* d_hist, int width, int height) {
    // Shared memory for per-block histogram
    __shared__ int sharedHist[NUM_BINS];

    // Initialize shared histogram bins to zero
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < NUM_BINS) {
        sharedHist[tid] = 0;
    }
    __syncthreads();

    // Compute pixel coordinates
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // Compute linear index in the global image
    if (x < width && y < height) {
        int idx = y * width + x;
        atomicAdd(&sharedHist[d_img[idx]], 1);
    }
    __syncthreads();

    // Reduce shared histogram into global histogram
    if (tid < NUM_BINS) {
        atomicAdd(&d_hist[tid], sharedHist[tid]);
    }
}


// CUDA kernel to apply the equalization using the precomputed lookup table (LUT).
__global__ void applyEqualization(const unsigned char* d_img, unsigned char* d_out, 
                                  const unsigned char* d_lut, int imgSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < imgSize) {
        d_out[idx] = d_lut[d_img[idx]];
    }
}

int processImageParallel(std::string path, std::string csvPath) {
    // Read the image using OpenCV.
    cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(input.empty()) {
        std::cerr << "Could not open or find the image at " << path << std::endl;
        return -1;
    }
    cv::resize(input, input, cv::Size(3840, 2160));  // Resize to 512x512 for consistency.
        
    // Get resolution and number of channels.
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    
    // Convert to grayscale if necessary since our CUDA kernel expects a single channel.
    cv::Mat gray;
    if(channels == 1) {
        gray = input;
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        channels = gray.channels();  // now should be 1
    }
    
    int imgSize = gray.cols * gray.rows;
    
    // Allocate host memory.
    unsigned char* h_img = (unsigned char*)malloc(imgSize * sizeof(unsigned char));
    unsigned char* h_out = (unsigned char*)malloc(imgSize * sizeof(unsigned char));
    int h_hist[NUM_BINS] = {0};
    unsigned char h_lut[NUM_BINS];
    
    // Copy image data from OpenCV Mat (assumed continuous) to host array.
    if (!gray.isContinuous()) {
        gray = gray.clone();
    }
    memcpy(h_img, gray.data, imgSize * sizeof(unsigned char));
    
    // Device memory allocation.
    unsigned char *d_img, *d_out, *d_lut;
    int* d_hist;
    cudaMalloc((void**)&d_img, imgSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int));
    cudaMalloc((void**)&d_lut, NUM_BINS * sizeof(unsigned char));
    
    // Copy the input image to the device and initialize the histogram memory to zero.
    cudaMemcpy(d_img, h_img, imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
    
    // Set up a 1D grid and block configuration.
    int threadsPerBlock = 256;
    int blocksPerGrid = (imgSize + threadsPerBlock - 1) / threadsPerBlock;
    
    // Start timing the execution.
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);  // 32x32 threads per block
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the tiled histogram kernel
    computeHistogramTiled<<<gridDim, blockDim>>>(d_img, d_hist, width, height);

    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << h_hist[i] << " ";
    }   

    
    // Copy the histogram data back to the host.
    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Compute the cumulative distribution function (CDF) on the host.
    int cdf[NUM_BINS] = {0};
    cdf[0] = h_hist[0];
    for (int i = 1; i < NUM_BINS; i++) {
        cdf[i] = cdf[i - 1] + h_hist[i];
    }
    
    // Find the first non-zero CDF value.
    int cdf_min = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if(cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // Create the lookup table (LUT) by normalizing the CDF.
    for (int i = 0; i < NUM_BINS; i++) {
        h_lut[i] = (unsigned char)(((float)(cdf[i] - cdf_min) / (imgSize - cdf_min)) * 255.0f + 0.5f);
    }
    
    // Copy the LUT to the device.
    cudaMemcpy(d_lut, h_lut, NUM_BINS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Launch the kernel to apply the equalization.
    applyEqualization<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_out, d_lut, imgSize);
    cudaDeviceSynchronize();
    
    // Stop timing.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execTime = end - start;
    
    // Copy the equalized image back to the host.
    cudaMemcpy(h_out, d_out, imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    
    
    // Create an OpenCV Mat from the equalized image data.
    cv::Mat equalizedImage(height, width, CV_8UC1, h_out);
    
    // Plot the equalized image (display in a window).
    cv::imshow("Equalized Image", equalizedImage);
    cv::waitKey(0);  // Wait indefinitely until a key is pressed.
    
    // Write statistics to CSV file in append mode.
    // CSV columns: Resolution,ExecutionTime (ms),Channels,Type
    std::ofstream csvFile(csvPath, std::ios::out | std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file at " << csvPath << std::endl;
        return -1;
    }
    // If file is empty, write header first.
    csvFile.seekp(0, std::ios::end);
    if (csvFile.tellp() == 0) {
        csvFile << "Resolution,ExecutionTime (ms),Channels,Type\n";
    }
    csvFile << std::to_string(width) + "x" + std::to_string(height) << ","
            << execTime.count() << "," << "PARALLEL" << ","
            << channels << "\n";
    csvFile.close();
    
    // Clean up device and host memory.
    free(h_img);
    // Note: h_out memory is now used by the cv::Mat, so clone if needed.
    // Alternatively, since we've displayed the image, we free h_out.
    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_lut);
    
    return 0;
}