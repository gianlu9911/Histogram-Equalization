#include "Kernels.cuh"

#define NUM_BINS 256
#define TILE_WIDTH 8

using namespace cv;
using namespace std;


// -----------------------------------------------------------------------------
// Helper function: Perform histogram equalization on device data.
// All operations (histogram, CDF, LUT computation, mapping) run on the GPU.
// -----------------------------------------------------------------------------
void histogramEqualization(unsigned char* d_in, unsigned char* d_out, int width, int height) {
    int imgSize = width * height;
    int *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
    
    // Launch kernel to compute histogram using tiling.
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    computeHistogramTiled<<<grid, block>>>(d_in, d_hist, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());


    
    // Copy histogram into a thrust device vector.
    thrust::device_vector<int> d_hist_vec(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_hist_vec.data()),
         d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToDevice));
    
    // Compute the CDF using an inclusive scan.
    thrust::device_vector<int> d_cdf(NUM_BINS);
    thrust::inclusive_scan(d_hist_vec.begin(), d_hist_vec.end(), d_cdf.begin());
    
    // Get the first value of the CDF (minimum non-zero value).
    int cdf0 = d_cdf.front();
    
    // Create a thrust device vector for the LUT.
    thrust::device_vector<unsigned char> d_lut(NUM_BINS);
    
    // Instead of using thrust::transform with a functor, we launch our kernel.
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_BINS + threadsPerBlock - 1) / threadsPerBlock;
    int* d_cdf_ptr = thrust::raw_pointer_cast(d_cdf.data());
    unsigned char* d_lut_ptr = thrust::raw_pointer_cast(d_lut.data());
    
    applyLutKernel<<<blocksPerGrid, threadsPerBlock>>>(d_cdf_ptr, d_lut_ptr, imgSize, cdf0, NUM_BINS);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Apply the equalization by mapping the original image through the LUT.
    int threads = 256;
    int blocks = (imgSize + threads - 1) / threads;
    applyEqualization<<<blocks, threads>>>(d_in, d_out, d_lut_ptr, imgSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_hist));
}


// -----------------------------------------------------------------------------
// Pipeline Function: Run both grayscale and RGB (Y-channel) equalization.
// Also record execution times and log them using the lambda below.
// -----------------------------------------------------------------------------
void runHistogramEqualizationPipelines(const Mat &grayImage, const Mat &rgbImage) {
    float grayscaleTime = 0.0f;
    float rgbTime = 0.0f;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------------- Grayscale Pipeline ----------------
    int graySize = grayImage.rows * grayImage.cols;
    unsigned char *d_gray = nullptr, *d_grayEqualized = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gray, graySize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_grayEqualized, graySize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_gray, grayImage.data, graySize * sizeof(unsigned char),
     cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start, 0));
    histogramEqualization(d_gray, d_grayEqualized, grayImage.cols, grayImage.rows);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&grayscaleTime, start, stop));
    
    Mat grayEqualized(grayImage.size(), grayImage.type());
    CUDA_CHECK(cudaMemcpy(grayEqualized.data, d_grayEqualized, graySize * 
        sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_grayEqualized));
    
    // ---------------- RGB Pipeline (Equalize Y Channel) ----------------
    int numPixels = rgbImage.rows * rgbImage.cols;
    // Pack the RGB image into uchar4 (assume input is in RGB order)
    uchar4 *h_rgb = new uchar4[numPixels];
    for (int i = 0; i < rgbImage.rows; i++) {
        for (int j = 0; j < rgbImage.cols; j++) {
            Vec3b pix = rgbImage.at<Vec3b>(i, j);
            h_rgb[i * rgbImage.cols + j] = make_uchar4(pix[0], pix[1], pix[2], 255);
        }
    }
    
    uchar4 *d_rgb = nullptr, *d_ycbcr = nullptr, *d_rgbEqualized = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, numPixels * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_ycbcr, numPixels * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_rgbEqualized, numPixels * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaEventRecord(start, 0));
    rgb2ycbcr<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_ycbcr, numPixels);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    unsigned char *d_Y = nullptr, *d_YEqualized = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Y, numPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_YEqualized, numPixels * sizeof(unsigned char)));
    extractYChannel<<<blocksPerGrid, threadsPerBlock>>>(d_ycbcr, d_Y, numPixels);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    histogramEqualization(d_Y, d_YEqualized, rgbImage.cols, rgbImage.rows);
    
    updateYChannel<<<blocksPerGrid, threadsPerBlock>>>(d_ycbcr, d_YEqualized, numPixels);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    ycbcr2rgb<<<blocksPerGrid, threadsPerBlock>>>(d_ycbcr, d_rgbEqualized, numPixels);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&rgbTime, start, stop));
    
    uchar4 *h_rgbEqualized = new uchar4[numPixels];
    CUDA_CHECK(cudaMemcpy(h_rgbEqualized, d_rgbEqualized, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
    
    Mat rgbEqualized(rgbImage.size(), CV_8UC3);
    for (int i = 0; i < rgbImage.rows; i++) {
        for (int j = 0; j < rgbImage.cols; j++) {
            uchar4 pix = h_rgbEqualized[i * rgbImage.cols + j];
            rgbEqualized.at<Vec3b>(i, j) = Vec3b(pix.x, pix.y, pix.z);
        }
    }
    
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_ycbcr));
    CUDA_CHECK(cudaFree(d_rgbEqualized));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_YEqualized));
    delete[] h_rgb;
    delete[] h_rgbEqualized;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Log execution times using a local lambda.
    auto writeCSVLine = [&](const std::string &line) {
        const std::string filePath = "../execution_times.csv";
        bool writeHeader = false;

        // Check if the file exists and is empty
        std::ifstream ifs(filePath);
        if (!ifs.good() || ifs.peek() == std::ifstream::traits_type::eof()) {
            writeHeader = true;
        }
        ifs.close();

        // Open in append mode and write header if needed
        std::ofstream ofs(filePath, std::ios::app);
        if (!ofs) {
            std::cerr << "Error opening CSV file." << std::endl;
            return;
        }

        if (writeHeader) {
            ofs << "Resolution,ExecutionTime,Mode,Channels\n";
        }

        ofs << line << "\n";
        ofs.close();
    };

    // Prepare and log the data
    std::string resolution = std::to_string(grayImage.cols) + "x" + std::to_string(grayImage.rows);
    std::string grayLine = resolution + "," + std::to_string(grayscaleTime) + ",PARALLEL,1";
    std::string rgbLine  = resolution + "," + std::to_string(rgbTime) + ",PARALLEL,3";

    writeCSVLine(grayLine);
    writeCSVLine(rgbLine);

    
    // Display results.
    imshow("Equalized Grayscale", grayEqualized);
    imwrite("lena_gray_equalized.png", grayEqualized);
    imwrite("lena_rgb_equalized.png", rgbEqualized);
    imshow("Equalized RGB (Y Channel Equalized)", rgbEqualized);
    waitKey(0);
}


// -----------------------------------------------------------------------------
// Main: Process all target resolutions using lambdas to increase reuse.
// -----------------------------------------------------------------------------
int processImageCuda(std::string imgPath) {
    // Warm-up: launch a dummy kernel to initialize GPU.
    dummyKernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Load the original RGB image.
    Mat originalRGB = imread(imgPath, IMREAD_COLOR);
    if (originalRGB.empty()) {
        cerr << "Error: Could not load RGB image!" << endl;
        return -1;
    }
    // Convert RGB to grayscale once.
    Mat originalGray;
    cvtColor(originalRGB, originalGray, COLOR_BGR2GRAY);
    
    // Define target resolutions: HD (1280x720), FullHD (1920x1080), 4K (3840x2160)
    vector<Size> resolutions = { Size(3840,2160), Size(1920,1080), Size(1280,720) };
    
    // Define a local lambda to process one resolution.
    auto processResolution = [&](const Size &res) {
        //cout << "Processing resolution: " << res.width << "x" << res.height << endl;
        Mat resizedGray, resizedRGB;
        resize(originalGray, resizedGray, res);
        resize(originalRGB, resizedRGB, res);
        runHistogramEqualizationPipelines(resizedGray, resizedRGB);
    };
    
    // Loop over each resolution.
    for (const Size &res : resolutions) {
        processResolution(res);
    }
    
    return 0;
}