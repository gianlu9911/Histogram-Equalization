#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void rgbToYCbCrKernelPitched(const uchar3* d_rgbImage, size_t pitch_rgb, 
                                        unsigned char* d_yImage, size_t pitch_y, 
                                        unsigned char* d_c1Image, size_t pitch_c1, 
                                        unsigned char* d_c2Image, size_t pitch_c2, 
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Use pitched memory addressing
        uchar3* row_rgb = (uchar3*)((char*)d_rgbImage + y * pitch_rgb);
        unsigned char* row_y = (unsigned char*)((char*)d_yImage + y * pitch_y);
        unsigned char* row_c1 = (unsigned char*)((char*)d_c1Image + y * pitch_c1);
        unsigned char* row_c2 = (unsigned char*)((char*)d_c2Image + y * pitch_c2);

        uchar3 rgb = row_rgb[x];

        float nR = rgb.x * 0.003921569f;
        float nG = rgb.y * 0.003921569f;
        float nB = rgb.z * 0.003921569f;

        float nY = 0.299f * nR + 0.587f * nG + 0.114f * nB;
        float nC1 = nB - nY;  
        float nC2 = nR - nY;  

        nC1 = 111.4f * 0.003921569f * nC1 + 156.0f * 0.003921569f;
        nC2 = 135.64f * 0.003921569f * nC2 + 137.0f * 0.003921569f;
        nY = 1.0f * 0.713267f * nY;

        row_y[x] = (unsigned char)(nY * 255.0f);
        row_c1[x] = (unsigned char)(nC1 * 255.0f);
        row_c2[x] = (unsigned char)(nC2 * 255.0f);
    }
}

void rgbToYCbCrPitched(const unsigned char* h_rgbImage, unsigned char* h_yImage, 
                       unsigned char* h_c1Image, unsigned char* h_c2Image, 
                       int width, int height) {
    size_t pitch_rgb, pitch_y, pitch_c1, pitch_c2;

    // Allocate pitched device memory
    uchar3* d_rgbImage;
    unsigned char* d_yImage;
    unsigned char* d_c1Image;
    unsigned char* d_c2Image;

    cudaMallocPitch(&d_rgbImage, &pitch_rgb, width * sizeof(uchar3), height);
    cudaMallocPitch(&d_yImage, &pitch_y, width * sizeof(unsigned char), height);
    cudaMallocPitch(&d_c1Image, &pitch_c1, width * sizeof(unsigned char), height);
    cudaMallocPitch(&d_c2Image, &pitch_c2, width * sizeof(unsigned char), height);

    // Copy data to device using cudaMemcpy2D
    cudaMemcpy2D(d_rgbImage, pitch_rgb, h_rgbImage, width * sizeof(uchar3), 
                 width * sizeof(uchar3), height, cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    rgbToYCbCrKernelPitched<<<gridSize, blockSize>>>(d_rgbImage, pitch_rgb, d_yImage, pitch_y, 
                                                     d_c1Image, pitch_c1, d_c2Image, pitch_c2, width, height);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy2D(h_yImage, width * sizeof(unsigned char), d_yImage, pitch_y, 
                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_c1Image, width * sizeof(unsigned char), d_c1Image, pitch_c1, 
                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_c2Image, width * sizeof(unsigned char), d_c2Image, pitch_c2, 
                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rgbImage);
    cudaFree(d_yImage);
    cudaFree(d_c1Image);
    cudaFree(d_c2Image);
}

int main() {
    cv::Mat img = cv::imread("../imgs/lena_rgb.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(3080, 2090));
    int width = img.cols;
    int height = img.rows;

    // Allocate pinned memory (page-locked memory)
    unsigned char* h_rgbImage;
    unsigned char* h_yImage;
    unsigned char* h_c1Image;
    unsigned char* h_c2Image;

    cudaMallocHost((void**)&h_rgbImage, width * height * 3); 
    cudaMallocHost((void**)&h_yImage, width * height);
    cudaMallocHost((void**)&h_c1Image, width * height);
    cudaMallocHost((void**)&h_c2Image, width * height);

    // Load image into pinned memory
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3b color = img.at<cv::Vec3b>(i, j);
            int idx = i * width + j;
            h_rgbImage[idx * 3] = color[2];  
            h_rgbImage[idx * 3 + 1] = color[1];
            h_rgbImage[idx * 3 + 2] = color[0];
        }
    }

    // Perform YCbCr conversion with pitched memory
    rgbToYCbCrPitched(h_rgbImage, h_yImage, h_c1Image, h_c2Image, width, height);

    // Save results
    cv::Mat yImage(height, width, CV_8UC1, h_yImage);
    cv::Mat c1Image(height, width, CV_8UC1, h_c1Image);
    cv::Mat c2Image(height, width, CV_8UC1, h_c2Image);

    cv::imwrite("y_channel.png", yImage);
    cv::imwrite("c1_channel.png", c1Image);
    cv::imwrite("c2_channel.png", c2Image);

    // Free pinned memory
    cudaFreeHost(h_rgbImage);
    cudaFreeHost(h_yImage);
    cudaFreeHost(h_c1Image);
    cudaFreeHost(h_c2Image);

    return 0;
}
