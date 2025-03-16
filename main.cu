#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void rgbToYCbCrKernelPitched(const uchar4* d_rgbImage, size_t pitch_rgb, 
                                        unsigned char* d_yImage, size_t pitch_y, 
                                        unsigned char* d_c1Image, size_t pitch_c1, 
                                        unsigned char* d_c2Image, size_t pitch_c2, 
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4* row_rgb = (uchar4*)((char*)d_rgbImage + y * pitch_rgb);
        unsigned char* row_y = (unsigned char*)((char*)d_yImage + y * pitch_y);
        unsigned char* row_c1 = (unsigned char*)((char*)d_c1Image + y * pitch_c1);
        unsigned char* row_c2 = (unsigned char*)((char*)d_c2Image + y * pitch_c2);

        uchar4 rgb = row_rgb[x];

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

float rgbToYCbCrPitched(const unsigned char* h_rgbImage, unsigned char* h_yImage, 
                       unsigned char* h_c1Image, unsigned char* h_c2Image, 
                       int width, int height) {
    size_t pitch_rgb, pitch_y, pitch_c1, pitch_c2;
    uchar4* d_rgbImage;
    unsigned char* d_yImage;
    unsigned char* d_c1Image;
    unsigned char* d_c2Image;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocPitch(&d_rgbImage, &pitch_rgb, width * sizeof(uchar4), height);
    cudaMallocPitch(&d_yImage, &pitch_y, width * sizeof(unsigned char), height);
    cudaMallocPitch(&d_c1Image, &pitch_c1, width * sizeof(unsigned char), height);
    cudaMallocPitch(&d_c2Image, &pitch_c2, width * sizeof(unsigned char), height);

    cudaMemcpy2DAsync(d_rgbImage, pitch_rgb, h_rgbImage, width * sizeof(uchar4), 
                      width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    rgbToYCbCrKernelPitched<<<gridSize, blockSize, 0, stream>>>(d_rgbImage, pitch_rgb, d_yImage, pitch_y, 
                                                                d_c1Image, pitch_c1, d_c2Image, pitch_c2, width, height);
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy2DAsync(h_yImage, width * sizeof(unsigned char), d_yImage, pitch_y, 
                      width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost, stream);
    cudaMemcpy2DAsync(h_c1Image, width * sizeof(unsigned char), d_c1Image, pitch_c1, 
                      width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost, stream);
    cudaMemcpy2DAsync(h_c2Image, width * sizeof(unsigned char), d_c2Image, pitch_c2, 
                      width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(d_rgbImage);
    cudaFree(d_yImage);
    cudaFree(d_c1Image);
    cudaFree(d_c2Image);

    return elapsedTime;
}

int main2() {
    cv::Mat img = cv::imread("../imgs/lena_rgb.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(3080, 2090));
    int width = img.cols;
    int height = img.rows;

    unsigned char* h_rgbImage;
    unsigned char* h_yImage;
    unsigned char* h_c1Image;
    unsigned char* h_c2Image;

    cudaMallocHost((void**)&h_rgbImage, width * height * 4); 
    cudaMallocHost((void**)&h_yImage, width * height);
    cudaMallocHost((void**)&h_c1Image, width * height);
    cudaMallocHost((void**)&h_c2Image, width * height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3b color = img.at<cv::Vec3b>(i, j);
            int idx = i * width + j;
            h_rgbImage[idx * 4] = color[2];  
            h_rgbImage[idx * 4 + 1] = color[1];
            h_rgbImage[idx * 4 + 2] = color[0];
            h_rgbImage[idx * 4 + 3] = 0;
        }
    }

    float executionTime = rgbToYCbCrPitched(h_rgbImage, h_yImage, h_c1Image, h_c2Image, width, height);
    std::cout << "Total execution time: " << executionTime << " ms" << std::endl;

    cv::Mat yImage(height, width, CV_8UC1, h_yImage);
    cv::Mat c1Image(height, width, CV_8UC1, h_c1Image);
    cv::Mat c2Image(height, width, CV_8UC1, h_c2Image);

    cv::imwrite("y_channel.png", yImage);
    cv::imwrite("c1_channel.png", c1Image);
    cv::imwrite("c2_channel.png", c2Image);

    cudaFreeHost(h_rgbImage);
    cudaFreeHost(h_yImage);
    cudaFreeHost(h_c1Image);
    cudaFreeHost(h_c2Image);



    return 0;
}


#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void yCbCrToRGBKernelPitched(const unsigned char* d_yImage, size_t pitch_y, 
    const unsigned char* d_c1Image, size_t pitch_c1, 
    const unsigned char* d_c2Image, size_t pitch_c2, 
    uchar4* d_rgbImage, size_t pitch_rgb, 
    int width, int height) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < width && y < height) {
unsigned char* row_y = (unsigned char*)((char*)d_yImage + y * pitch_y);
unsigned char* row_c1 = (unsigned char*)((char*)d_c1Image + y * pitch_c1);
unsigned char* row_c2 = (unsigned char*)((char*)d_c2Image + y * pitch_c2);
uchar4* row_rgb = (uchar4*)((char*)d_rgbImage + y * pitch_rgb);

float Y = row_y[x];
float Cb = row_c1[x] - 128.0f;
float Cr = row_c2[x] - 128.0f;

float R = Y + 1.402f * Cr;
float G = Y - 0.344136f * Cb - 0.714136f * Cr;
float B = Y + 1.772f * Cb;

row_rgb[x].x = min(max((int)B, 0), 255);
row_rgb[x].y = min(max((int)G, 0), 255);
row_rgb[x].z = min(max((int)R, 0), 255);
row_rgb[x].w = 255;
}
}
float yCbCrToRGBPitched(unsigned char* h_yImage, unsigned char* h_c1Image, unsigned char* h_c2Image, 
    unsigned char* h_rgbImage, int width, int height) {
size_t pitch_y, pitch_c1, pitch_c2, pitch_rgb;
unsigned char* d_yImage;
unsigned char* d_c1Image;
unsigned char* d_c2Image;
uchar4* d_rgbImage;
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMallocPitch(&d_yImage, &pitch_y, width * sizeof(unsigned char), height);
cudaMallocPitch(&d_c1Image, &pitch_c1, width * sizeof(unsigned char), height);
cudaMallocPitch(&d_c2Image, &pitch_c2, width * sizeof(unsigned char), height);
cudaMallocPitch(&d_rgbImage, &pitch_rgb, width * sizeof(uchar4), height);

cudaMemcpy2DAsync(d_yImage, pitch_y, h_yImage, width * sizeof(unsigned char), 
  width * sizeof(unsigned char), height, cudaMemcpyHostToDevice, stream);
cudaMemcpy2DAsync(d_c1Image, pitch_c1, h_c1Image, width * sizeof(unsigned char), 
  width * sizeof(unsigned char), height, cudaMemcpyHostToDevice, stream);
cudaMemcpy2DAsync(d_c2Image, pitch_c2, h_c2Image, width * sizeof(unsigned char), 
  width * sizeof(unsigned char), height, cudaMemcpyHostToDevice, stream);

dim3 blockSize(16, 16);
dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
(height + blockSize.y - 1) / blockSize.y);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);

yCbCrToRGBKernelPitched<<<gridSize, blockSize, 0, stream>>>(d_yImage, pitch_y, d_c1Image, pitch_c1, 
                                            d_c2Image, pitch_c2, d_rgbImage, pitch_rgb, width, height);
cudaStreamSynchronize(stream);

cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);

cudaMemcpy2DAsync(h_rgbImage, width * sizeof(uchar4), d_rgbImage, pitch_rgb, 
  width * sizeof(uchar4), height, cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);

cudaFree(d_yImage);
cudaFree(d_c1Image);
cudaFree(d_c2Image);
cudaFree(d_rgbImage);

return elapsedTime;
}
int main() {
    cv::Mat img = cv::imread("../imgs/lena_rgb.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(3200, 3200));
    int width = img.cols;
    int height = img.rows;

    unsigned char* h_rgbImage;
    unsigned char* h_yImage;
    unsigned char* h_c1Image;
    unsigned char* h_c2Image;
    unsigned char* h_reconstructedRGB;

    cudaMallocHost((void**)&h_rgbImage, width * height * 4);
    cudaMallocHost((void**)&h_yImage, width * height);
    cudaMallocHost((void**)&h_c1Image, width * height);
    cudaMallocHost((void**)&h_c2Image, width * height);
    cudaMallocHost((void**)&h_reconstructedRGB, width * height * 4);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3b color = img.at<cv::Vec3b>(i, j);
            int idx = i * width + j;
            h_rgbImage[idx * 4] = color[2];  
            h_rgbImage[idx * 4 + 1] = color[1];
            h_rgbImage[idx * 4 + 2] = color[0];
            h_rgbImage[idx * 4 + 3] = 255;
        }
    }

    float execTimeYCbCr = rgbToYCbCrPitched(h_rgbImage, h_yImage, h_c1Image, h_c2Image, width, height);
    std::cout << "YCbCr conversion time: " << execTimeYCbCr << " ms" << std::endl;

    float execTimeRGB = yCbCrToRGBPitched(h_yImage, h_c1Image, h_c2Image, h_reconstructedRGB, width, height);
    std::cout << "RGB reconstruction time: " << execTimeRGB << " ms" << std::endl;

    cv::Mat reconstructedImg(height, width, CV_8UC4, h_reconstructedRGB);
    cv::imwrite("reconstructed_rgb.png", reconstructedImg);

    cudaFreeHost(h_rgbImage);
    cudaFreeHost(h_yImage);
    cudaFreeHost(h_c1Image);
    cudaFreeHost(h_c2Image);
    cudaFreeHost(h_reconstructedRGB);

    return 0;
}

