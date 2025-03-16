#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define NUM_BINS 256
#define TILE_WIDTH 32

// -----------------------------------------------------------------------------
// CUDA Error Checking Macro
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------------------------
// CUDA Kernels
// -----------------------------------------------------------------------------

// Grayscale histogram kernel using tiling and shared memory.
__global__ void computeHistogramTiled(const unsigned char* d_img, int* d_hist, int width, int height) {
    __shared__ int sharedHist[NUM_BINS];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < NUM_BINS)
        sharedHist[tid] = 0;
    __syncthreads();

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        atomicAdd(&sharedHist[d_img[idx]], 1);
    }
    __syncthreads();
    if (tid < NUM_BINS)
        atomicAdd(&d_hist[tid], sharedHist[tid]);
}

// Kernel to apply histogram equalization using a precomputed lookup table.
__global__ void applyEqualization(const unsigned char* d_img, unsigned char* d_out, 
                                  const unsigned char* d_lut, int imgSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < imgSize)
        d_out[idx] = d_lut[d_img[idx]];
}

// Kernel for converting RGBA image to YCbCr using pitched memory.
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
        nY  = 1.0f * 0.713267f * nY;
        row_y[x]  = (unsigned char)(nY * 255.0f);
        row_c1[x] = (unsigned char)(nC1 * 255.0f);
        row_c2[x] = (unsigned char)(nC2 * 255.0f);
    }
}

// Kernel for converting YCbCr back to RGBA using pitched memory.
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

// -----------------------------------------------------------------------------
// Pipeline Functions
// -----------------------------------------------------------------------------

// Color Pipeline: Convert RGB image to YCbCr, equalize the Y channel, and convert back to RGB.
float runColorPipelineWithEqualization(const std::string& path) {
    // Step 1: Read and resize the RGB image.
    cv::Mat colorImg = cv::imread(path, cv::IMREAD_COLOR);
    if (colorImg.empty()) {
        std::cerr << "Error: Could not load color image: " << path << std::endl;
        return -1;
    }
    cv::resize(colorImg, colorImg, cv::Size(3080, 2090));
    int width = colorImg.cols, height = colorImg.rows;

    // Step 2: Convert BGR to BGRA.
    cv::Mat rgbaImg;
    cv::cvtColor(colorImg, rgbaImg, cv::COLOR_BGR2BGRA);

    // Step 3: Allocate pinned host memory for RGBA image and YCbCr channels.
    unsigned char *h_rgbImage, *h_yImage, *h_c1Image, *h_c2Image;
    unsigned char *h_reconstructedRGB;
    CUDA_CHECK(cudaMallocHost((void**)&h_rgbImage, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMallocHost((void**)&h_yImage, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocHost((void**)&h_c1Image, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocHost((void**)&h_c2Image, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocHost((void**)&h_reconstructedRGB, width * height * sizeof(uchar4)));

    memcpy(h_rgbImage, rgbaImg.data, width * height * sizeof(uchar4));

    // Step 4: Convert RGB to YCbCr on GPU.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uchar4* d_rgbImage;
    unsigned char *d_yImage, *d_c1Image, *d_c2Image;
    size_t pitch_rgb, pitch_y, pitch_c1, pitch_c2;
    CUDA_CHECK(cudaMallocPitch(&d_rgbImage, &pitch_rgb, width * sizeof(uchar4), height));
    CUDA_CHECK(cudaMallocPitch(&d_yImage, &pitch_y, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMallocPitch(&d_c1Image, &pitch_c1, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMallocPitch(&d_c2Image, &pitch_c2, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMemcpy2DAsync(d_rgbImage, pitch_rgb, h_rgbImage, width * sizeof(uchar4),
                                 width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream));
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    rgbToYCbCrKernelPitched<<<gridSize, blockSize, 0, stream>>>(d_rgbImage, pitch_rgb, d_yImage, pitch_y, 
                                                                 d_c1Image, pitch_c1, d_c2Image, pitch_c2, width, height);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy2DAsync(h_yImage, width * sizeof(unsigned char), d_yImage, pitch_y,
                                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(h_c1Image, width * sizeof(unsigned char), d_c1Image, pitch_c1,
                                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(h_c2Image, width * sizeof(unsigned char), d_c2Image, pitch_c2,
                                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 5: Equalize the Y channel using the grayscale pipeline.
    unsigned char* h_yEqualized;
    CUDA_CHECK(cudaMallocHost((void**)&h_yEqualized, width * height * sizeof(unsigned char)));
    unsigned char *d_y, *d_yOut, *d_yLut;
    int *d_yHist;
    CUDA_CHECK(cudaMalloc(&d_y, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_yOut, width * height * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_yHist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_yLut, NUM_BINS * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_y, h_yImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_yHist, 0, NUM_BINS * sizeof(int)));
    dim3 grayBlockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 grayGridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    computeHistogramTiled<<<grayGridDim, grayBlockDim>>>(d_y, d_yHist, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_yHist[NUM_BINS] = {0};
    CUDA_CHECK(cudaMemcpy(h_yHist, d_yHist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    int cdf[NUM_BINS] = {0};
    cdf[0] = h_yHist[0];
    for (int i = 1; i < NUM_BINS; i++)
        cdf[i] = cdf[i - 1] + h_yHist[i];
    int cdf_min = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (cdf[i] != 0) { cdf_min = cdf[i]; break; }
    }
    unsigned char h_yLut[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        h_yLut[i] = (unsigned char)(((float)(cdf[i] - cdf_min) / (width * height - cdf_min)) * 255.0f + 0.5f);
    }
    CUDA_CHECK(cudaMemcpy(d_yLut, h_yLut, NUM_BINS * sizeof(unsigned char), cudaMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    applyEqualization<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_yOut, d_yLut, width * height);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_yEqualized, d_yOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_yOut)); CUDA_CHECK(cudaFree(d_yHist)); CUDA_CHECK(cudaFree(d_yLut));
    memcpy(h_yImage, h_yEqualized, width * height * sizeof(unsigned char));
    CUDA_CHECK(cudaFreeHost(h_yEqualized));

    // Step 6: Recombine equalized Y with original Cb and Cr and convert back to RGB.
    unsigned char *d_y2, *d_c12, *d_c22;
    uchar4* d_rgbOut;
    size_t pitch_y2, pitch_c12, pitch_c22, pitch_rgbOut;
    CUDA_CHECK(cudaMallocPitch(&d_y2, &pitch_y2, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMallocPitch(&d_c12, &pitch_c12, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMallocPitch(&d_c22, &pitch_c22, width * sizeof(unsigned char), height));
    CUDA_CHECK(cudaMallocPitch(&d_rgbOut, &pitch_rgbOut, width * sizeof(uchar4), height));
    CUDA_CHECK(cudaMemcpy2D(d_y2, pitch_y2, h_yImage, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_c12, pitch_c12, h_c1Image, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_c22, pitch_c22, h_c2Image, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice));
    yCbCrToRGBKernelPitched<<<gridSize, blockSize>>>(d_y2, pitch_y2, d_c12, pitch_c12, d_c22, pitch_c22, d_rgbOut, pitch_rgbOut, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned char* h_finalRGB;
    CUDA_CHECK(cudaMallocHost(&h_finalRGB, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy2D(h_finalRGB, width * sizeof(uchar4), d_rgbOut, pitch_rgbOut, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost));
    cv::Mat finalImg(height, width, CV_8UC4, h_finalRGB);
    cv::imshow("Final Equalized RGB Image", finalImg);
    cv::imwrite("final_equalized_rgb.png", finalImg);
    cv::waitKey(0);

    // Step 7: Cleanup.
    CUDA_CHECK(cudaFree(d_rgbImage));
    CUDA_CHECK(cudaFree(d_yImage));
    CUDA_CHECK(cudaFree(d_c1Image));
    CUDA_CHECK(cudaFree(d_c2Image));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaFree(d_c12));
    CUDA_CHECK(cudaFree(d_c22));
    CUDA_CHECK(cudaFree(d_rgbOut));
    CUDA_CHECK(cudaFreeHost(h_rgbImage));
    CUDA_CHECK(cudaFreeHost(h_yImage));
    CUDA_CHECK(cudaFreeHost(h_c1Image));
    CUDA_CHECK(cudaFreeHost(h_c2Image));
    CUDA_CHECK(cudaFreeHost(h_finalRGB));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}

// Grayscale Pipeline: Run histogram equalization on a grayscale image.
double runGrayscalePipeline(const std::string& path, const std::string& csvPath) {
    cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (input.empty()){
        std::cerr << "Error: Could not load grayscale image: " << path << std::endl;
        return -1;
    }
    cv::resize(input, input, cv::Size(500,500));
    int grayWidth = input.cols, grayHeight = input.rows;
    int imgSize = grayWidth * grayHeight;

    unsigned char *h_img, *h_out;
    CUDA_CHECK(cudaMallocHost((void**)&h_img, imgSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, imgSize * sizeof(unsigned char)));
    int h_hist[NUM_BINS] = {0};
    unsigned char h_lut[NUM_BINS];

    if (!input.isContinuous())
        input = input.clone();
    memcpy(h_img, input.data, imgSize * sizeof(unsigned char));

    unsigned char *d_img, *d_out, *d_lut;
    int *d_hist;
    CUDA_CHECK(cudaMalloc((void**)&d_img, imgSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, imgSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lut, NUM_BINS * sizeof(unsigned char)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    size_t pitch = grayWidth * sizeof(unsigned char);
    CUDA_CHECK(cudaMemcpy2DAsync(d_img, pitch, h_img, pitch, pitch, grayHeight, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, NUM_BINS * sizeof(int), stream));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((grayWidth + TILE_WIDTH - 1) / TILE_WIDTH,
                 (grayHeight + TILE_WIDTH - 1) / TILE_WIDTH);

    auto startTime = std::chrono::high_resolution_clock::now();
    computeHistogramTiled<<<gridDim, blockDim, 0, stream>>>(d_img, d_hist, grayWidth, grayHeight);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int cdf[NUM_BINS] = {0};
    cdf[0] = h_hist[0];
    for (int i = 1; i < NUM_BINS; i++)
        cdf[i] = cdf[i - 1] + h_hist[i];
    int cdf_min = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (cdf[i] != 0) { cdf_min = cdf[i]; break; }
    }
    for (int i = 0; i < NUM_BINS; i++)
        h_lut[i] = (unsigned char)(((float)(cdf[i] - cdf_min) / (imgSize - cdf_min)) * 255.0f + 0.5f);
    CUDA_CHECK(cudaMemcpyAsync(d_lut, h_lut, NUM_BINS * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;
    int blocksPerGrid = (imgSize + threadsPerBlock - 1) / threadsPerBlock;
    applyEqualization<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_img, d_out, d_lut, imgSize);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execTime = endTime - startTime;

    CUDA_CHECK(cudaMemcpy2DAsync(h_out, pitch, d_out, pitch, pitch, grayHeight, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cv::Mat equalizedImage(grayHeight, grayWidth, CV_8UC1, h_out);
    cv::imshow("Original Image", input);
    cv::imshow("Equalized Image", equalizedImage);
    cv::waitKey(0);

    std::ofstream csvFile(csvPath, std::ios::out | std::ios::app);
    if (!csvFile.is_open())
        std::cerr << "Error opening CSV file: " << csvPath << std::endl;
    if (csvFile.tellp() == 0)
        csvFile << "Resolution,ExecutionTime (ms),Channels,Type\n";
    csvFile << std::to_string(grayWidth) + "x" + std::to_string(grayHeight) << ","
            << execTime.count() << ",PARALLEL,1\n";
    csvFile.close();

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_lut));
    CUDA_CHECK(cudaFreeHost(h_img));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return execTime.count();
}

// -----------------------------------------------------------------------------
// Combined processing function: Run both pipelines.
// -----------------------------------------------------------------------------
int processImageParallelCombined(const std::string& path, const std::string& csvPath) {
    std::cout << "Running Color Pipeline with Y-Channel Equalization:" << std::endl;
    //float colorTime = runColorPipelineWithEqualization(path);
    //std::cout << "Total Color Pipeline Time: " << colorTime << " ms" << std::endl;

    std::cout << "Running Grayscale Pipeline:" << std::endl;
    double grayTime = runGrayscalePipeline(path, csvPath);
    std::cout << "Total Grayscale Pipeline Time: " << grayTime << " ms" << std::endl;

    return 0;
}

