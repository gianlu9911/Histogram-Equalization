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

int processImageParallelCombined(std::string path, std::string csvPath) {
    // ---------------------------
    // Color Pipeline (YCbCr conversion)
    // ---------------------------
    cv::Mat colorImg = cv::imread(path, cv::IMREAD_COLOR);
    if(colorImg.empty()){
        std::cerr << "Error: Could not load image: " << path << std::endl;
        return -1;
    }
    // Resize to a desired resolution (example: 3080x2090)
    cv::resize(colorImg, colorImg, cv::Size(3080, 2090));
    int width = colorImg.cols, height = colorImg.rows;

    // Allocate pinned host memory for an RGBA image and YCbCr channels.
    unsigned char *h_rgbImage = nullptr, *h_yImage = nullptr;
    unsigned char *h_c1Image = nullptr, *h_c2Image = nullptr;
    unsigned char *h_reconstructedRGB = nullptr;
    cudaMallocHost((void**)&h_rgbImage, width * height * 4);
    cudaMallocHost((void**)&h_yImage, width * height);
    cudaMallocHost((void**)&h_c1Image, width * height);
    cudaMallocHost((void**)&h_c2Image, width * height);
    cudaMallocHost((void**)&h_reconstructedRGB, width * height * 4);

    // Copy the image from cv::Mat (BGR) into our RGBA pinned memory.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3b color = colorImg.at<cv::Vec3b>(i, j);
            int idx = i * width + j;
            h_rgbImage[idx * 4]     = color[2];  // R
            h_rgbImage[idx * 4 + 1] = color[1];  // G
            h_rgbImage[idx * 4 + 2] = color[0];  // B
            h_rgbImage[idx * 4 + 3] = 255;       // A
        }
    }
    // Convert RGB to YCbCr (device function using optimized transfers)
    float execTimeYCbCr = rgbToYCbCrPitched(h_rgbImage, h_yImage, h_c1Image, h_c2Image, width, height);
    std::cout << "YCbCr conversion time: " << execTimeYCbCr << " ms" << std::endl;
    
    // Reconstruct RGB from YCbCr
    float execTimeRGB = yCbCrToRGBPitched(h_yImage, h_c1Image, h_c2Image, h_reconstructedRGB, width, height);
    std::cout << "RGB reconstruction time: " << execTimeRGB << " ms" << std::endl;
    
    cv::Mat reconstructedImg(height, width, CV_8UC4, h_reconstructedRGB);
    cv::imwrite("reconstructed_rgb.png", reconstructedImg);

    // Free color-pipeline pinned memory.
    cudaFreeHost(h_rgbImage);
    cudaFreeHost(h_yImage);
    cudaFreeHost(h_c1Image);
    cudaFreeHost(h_c2Image);
    cudaFreeHost(h_reconstructedRGB);

    // ---------------------------
    // Grayscale Pipeline (Histogram Equalization)
    // ---------------------------
    cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(input.empty()){
        std::cerr << "Error: Could not open or find image: " << path << std::endl;
        return -1;
    }
    // Resize to 3840x2160 for this pipeline.
    cv::resize(input, input, cv::Size(3840, 2160));
    int grayWidth = input.cols, grayHeight = input.rows;
    int imgSize = grayWidth * grayHeight;

    // Allocate pinned host memory for the grayscale image and output.
    unsigned char *h_img = nullptr, *h_out = nullptr;
    cudaMallocHost((void**)&h_img, imgSize * sizeof(unsigned char));
    cudaMallocHost((void**)&h_out, imgSize * sizeof(unsigned char));
    int h_hist[NUM_BINS] = {0};
    unsigned char h_lut[NUM_BINS];

    if(!input.isContinuous())
        input = input.clone();
    memcpy(h_img, input.data, imgSize * sizeof(unsigned char));

    // Allocate device memory.
    unsigned char *d_img = nullptr, *d_out = nullptr, *d_lut = nullptr;
    int *d_hist = nullptr;
    cudaMalloc((void**)&d_img, imgSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int));
    cudaMalloc((void**)&d_lut, NUM_BINS * sizeof(unsigned char));

    // Create a CUDA stream for asynchronous transfers.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t pitch = grayWidth * sizeof(unsigned char);
    cudaMemcpy2DAsync(d_img, pitch, h_img, pitch, pitch, grayHeight, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_hist, 0, NUM_BINS * sizeof(int), stream);

    // Launch the tiled histogram kernel.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((grayWidth + TILE_WIDTH - 1) / TILE_WIDTH,
                 (grayHeight + TILE_WIDTH - 1) / TILE_WIDTH);
    
    auto start = std::chrono::high_resolution_clock::now();
    computeHistogramTiled<<<gridDim, blockDim, 0, stream>>>(d_img, d_hist, grayWidth, grayHeight);
    cudaStreamSynchronize(stream);
    
    // Asynchronously copy histogram back to host.
    cudaMemcpyAsync(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Compute the CDF and LUT.
    int cdf[NUM_BINS] = {0};
    cdf[0] = h_hist[0];
    for (int i = 1; i < NUM_BINS; i++) {
        cdf[i] = cdf[i - 1] + h_hist[i];
    }
    int cdf_min = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    for (int i = 0; i < NUM_BINS; i++) {
        h_lut[i] = (unsigned char)(((float)(cdf[i] - cdf_min) / (imgSize - cdf_min)) * 255.0f + 0.5f);
    }
    cudaMemcpyAsync(d_lut, h_lut, NUM_BINS * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
    
    // Launch the equalization kernel.
    int threadsPerBlock = 256;
    int blocksPerGrid = (imgSize + threadsPerBlock - 1) / threadsPerBlock;
    applyEqualization<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_img, d_out, d_lut, imgSize);
    cudaStreamSynchronize(stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execTime = end - start;
    
    cudaMemcpy2DAsync(h_out, pitch, d_out, pitch, pitch, grayHeight, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat equalizedImage(grayHeight, grayWidth, CV_8UC1, h_out);
    cv::imshow("Equalized Image", equalizedImage);
    cv::waitKey(0);

    // Append statistics to CSV.
    std::ofstream csvFile(csvPath, std::ios::out | std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file: " << csvPath << std::endl;
        return -1;
    }
    csvFile.seekp(0, std::ios::end);
    if (csvFile.tellp() == 0) {
        csvFile << "Resolution,ExecutionTime (ms),Channels,Type\n";
    }
    csvFile << std::to_string(grayWidth) + "x" + std::to_string(grayHeight) << ","
            << execTime.count() << "," << "PARALLEL" << "," << 1 << "\n";
    csvFile.close();

    // Clean up device memory.
    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_lut);

    // Free pinned host memory.
    cudaFreeHost(h_img);
    cudaFreeHost(h_out);
    cudaStreamDestroy(stream);

    return 0;
}