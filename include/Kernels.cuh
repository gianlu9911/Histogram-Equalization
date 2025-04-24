#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <vector>

#define NUM_BINS 256
#define TILE_WIDTH 8

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
// Dummy kernel for warm-up
// -----------------------------------------------------------------------------
__global__ void dummyKernel() {
    // Do nothing; just a simple kernel launch.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < 1)
        ;
}

// -----------------------------------------------------------------------------
// Kernel Definitions
// -----------------------------------------------------------------------------

__global__ void computeHistogramTiled(const unsigned char* d_img, int* d_hist, int width, int height) {
    __shared__ int sharedHist[NUM_BINS];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // Initialize the shared histogram in parallel
    for (int i = tid; i < NUM_BINS; i += blockSize) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Compute global pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute histogram in shared memory
    if (x < width && y < height) {
        int idx = y * width + x;
        atomicAdd(&sharedHist[d_img[idx]], 1);
    }
    __syncthreads();

    // Merge shared histogram to global histogram
    for (int i = tid; i < NUM_BINS; i += blockSize) {
        atomicAdd(&d_hist[i], sharedHist[i]);
    }
}



// Kernel to apply histogram equalization using a precomputed lookup table.
__global__ void applyEqualization(const unsigned char* d_img, unsigned char* d_out, const unsigned char* d_lut, int imgSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < imgSize)
        d_out[idx] = d_lut[d_img[idx]];
}

// CUDA kernel to convert RGB to YCbCr using vectorized (uchar4) operations.
__global__ void rgb2ycbcr(const uchar4* d_rgb, uchar4* d_ycbcr, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        uchar4 pix = d_rgb[idx];
        float r = static_cast<float>(pix.x);
        float g = static_cast<float>(pix.y);
        float b = static_cast<float>(pix.z);
        float y  =  0.299f * r + 0.587f * g + 0.114f * b;
        float cb = -0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f;
        float cr =  0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f;
        d_ycbcr[idx] = make_uchar4(static_cast<unsigned char>(y),
                                   static_cast<unsigned char>(cb),
                                   static_cast<unsigned char>(cr),
                                   pix.w);
    }
}

// CUDA kernel to convert YCbCr to RGB using vectorized (uchar4) operations.

__global__ void ycbcr2rgb(const uchar4* d_ycbcr, uchar4* d_rgb, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        uchar4 pix = d_ycbcr[idx];
        float y  = static_cast<float>(pix.x);
        float cb = static_cast<float>(pix.y);
        float cr = static_cast<float>(pix.z);
        float r = y + 1.402f * (cr - 128.0f);
        float g = y - 0.344136f * (cb - 128.0f) - 0.714136f * (cr - 128.0f);
        float b = y + 1.772f * (cb - 128.0f);
        unsigned char R = static_cast<unsigned char>(min(max(r, 0.0f), 255.0f));
        unsigned char G = static_cast<unsigned char>(min(max(g, 0.0f), 255.0f));
        unsigned char B = static_cast<unsigned char>(min(max(b, 0.0f), 255.0f));
        d_rgb[idx] = make_uchar4(R, G, B, pix.w);
    }
}

// Kernel to extract the Y channel from a YCbCr image.
__global__ void extractYChannel(const uchar4* d_ycbcr, unsigned char* d_Y, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels)
         d_Y[idx] = d_ycbcr[idx].x;
}

// Kernel to update the Y channel in a YCbCr image.
__global__ void updateYChannel(uchar4* d_ycbcr, const unsigned char* d_YEqualized, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
         uchar4 pix = d_ycbcr[idx];
         pix.x = d_YEqualized[idx];
         d_ycbcr[idx] = pix;
    }
}

// Device function to compute a single LUT value from a CDF value.
__device__ unsigned char computeLutValue(int cdf_val, int totalPixels, int cdf0) {
    float norm = (cdf_val - cdf0) / float(totalPixels - cdf0);
    return static_cast<unsigned char>(norm * 255.0f + 0.5f);
}

// Kernel that applies the computeLutValue function to each element of the CDF array.
__global__ void applyLutKernel(const int* d_cdf, unsigned char* d_lut, int totalPixels, int cdf0, int numBins) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numBins) {
        d_lut[idx] = computeLutValue(d_cdf[idx], totalPixels, cdf0);
    }
}
