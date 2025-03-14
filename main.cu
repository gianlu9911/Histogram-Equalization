#include "Sequential.h"
#include "Parallel.cuh"
int main2() {
    // Update the paths as needed
    string path = "../imgs/lena_4k.jpg";
    string csvPath = "../execution_times.csv";

    processImage(path, csvPath);
    processImageParallel(path, csvPath);

    return 0;
}


#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>  // Include OpenCV for image loading and manipulation

struct Pixel {
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

__global__ void rgbToYCbCrKernel(const Pixel* d_rgbImage, unsigned char* d_yImage, unsigned char* d_c1Image, unsigned char* d_c2Image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelIndex = y * width + x;  // Index for the pixel

        // Fetch RGB values from coalesced memory (using struct)
        unsigned char R = d_rgbImage[pixelIndex].R;
        unsigned char G = d_rgbImage[pixelIndex].G;
        unsigned char B = d_rgbImage[pixelIndex].B;

        // Convert to normalized floats
        float nNormalizedR = R * 0.003921569f;
        float nNormalizedG = G * 0.003921569f;
        float nNormalizedB = B * 0.003921569f;

        // Compute YCbCr components
        float nY = 0.299f * nNormalizedR + 0.587f * nNormalizedG + 0.114f * nNormalizedB;
        float nC1 = nNormalizedB - nY;  // Cb
        nC1 = 111.4f * 0.003921569f * nC1 + 156.0f * 0.003921569f;
        float nC2 = nNormalizedR - nY;  // Cr
        nC2 = 135.64f * 0.003921569f * nC2 + 137.0f * 0.003921569f;

        // Adjust the Y component
        nY = 1.0f * 0.713267f * nY;

        // Write back the YCbCr components
        d_yImage[pixelIndex] = (unsigned char)(nY * 255.0f);
        d_c1Image[pixelIndex] = (unsigned char)(nC1 * 255.0f);  // Cb
        d_c2Image[pixelIndex] = (unsigned char)(nC2 * 255.0f);  // Cr
    }
}

void rgbToYCbCr(const unsigned char* h_rgbImage, unsigned char* h_yImage, unsigned char* h_c1Image, unsigned char* h_c2Image, int width, int height) {
    // Number of pixels
    int numPixels = width * height;

    // Allocate device memory
    Pixel* d_rgbImage;
    unsigned char* d_yImage;
    unsigned char* d_c1Image;
    unsigned char* d_c2Image;

    cudaMalloc((void**)&d_rgbImage, numPixels * sizeof(Pixel));
    cudaMalloc((void**)&d_yImage, numPixels * sizeof(unsigned char));
    cudaMalloc((void**)&d_c1Image, numPixels * sizeof(unsigned char));
    cudaMalloc((void**)&d_c2Image, numPixels * sizeof(unsigned char));

    // Copy data to device
    cudaMemcpy(d_rgbImage, h_rgbImage, numPixels * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    rgbToYCbCrKernel<<<gridSize, blockSize>>>(d_rgbImage, d_yImage, d_c1Image, d_c2Image, width, height);

    // Check for kernel launch errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copy results back to host
    cudaMemcpy(h_yImage, d_yImage, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c1Image, d_c1Image, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2Image, d_c2Image, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rgbImage);
    cudaFree(d_yImage);
    cudaFree(d_c1Image);
    cudaFree(d_c2Image);
}

int main() {
    // Read the image using OpenCV
    cv::Mat img = cv::imread("../imgs/lena_rgb.png", cv::IMREAD_COLOR);  // Read the image in RGB format
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(3080,2090));  // Resize to 512x512 for consistency
    int width = img.cols;  // Image width
    int height = img.rows; // Image height

    // Allocate host memory for RGB and YCbCr channels
    unsigned char* h_rgbImage = new unsigned char[width * height * 3]; // RGB image data
    unsigned char* h_yImage = new unsigned char[width * height];        // Y channel
    unsigned char* h_c1Image = new unsigned char[width * height];       // Cb channel
    unsigned char* h_c2Image = new unsigned char[width * height];       // Cr channel

    // Convert OpenCV image (Mat) to raw RGB data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3b color = img.at<cv::Vec3b>(i, j);  // BGR format in OpenCV
            int idx = (i * width + j) * 3;
            h_rgbImage[idx] = color[2]; // R
            h_rgbImage[idx + 1] = color[1]; // G
            h_rgbImage[idx + 2] = color[0]; // B
        }
    }

    // Convert RGB to YCbCr
    rgbToYCbCr(h_rgbImage, h_yImage, h_c1Image, h_c2Image, width, height);

    // For example, save the Y, Cb, and Cr channels as images (if needed)
    cv::Mat yImage(height, width, CV_8UC1, h_yImage);
    cv::Mat c1Image(height, width, CV_8UC1, h_c1Image);
    cv::Mat c2Image(height, width, CV_8UC1, h_c2Image);
    
    cv::imwrite("y_channel.png", yImage);
    cv::imwrite("c1_channel.png", c1Image); // Cb
    cv::imwrite("c2_channel.png", c2Image); // Cr

    // Clean up
    delete[] h_rgbImage;
    delete[] h_yImage;
    delete[] h_c1Image;
    delete[] h_c2Image;

    return 0;
}