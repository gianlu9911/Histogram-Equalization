#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>  // for memset

using namespace std;
using namespace cv;

#define L 256  // Number of intensity levels

// Function to compute histogram for a single channel image
void computeHistogram(const Mat &channel, int histogram[L]) {
    memset(histogram, 0, L * sizeof(int)); // Initialize histogram to 0
    for (int i = 0; i < channel.rows; i++) {
        for (int j = 0; j < channel.cols; j++) {
            histogram[channel.at<uchar>(i, j)]++;
        }
    }
}

// Function to compute cumulative distribution function (CDF) from histogram
void computeCDF(const int histogram[L], float cdf[L], int totalPixels) {
    cdf[0] = static_cast<float>(histogram[0]) / totalPixels;
    for (int i = 1; i < L; i++) {
        cdf[i] = cdf[i - 1] + static_cast<float>(histogram[i]) / totalPixels;
    }
}

// Function to perform histogram equalization on a single channel using the computed CDF
void performHistogramEqualization(Mat &channel) {
    int histogram[L];
    float cdf[L];

    // Compute the histogram and CDF
    computeHistogram(channel, histogram);
    computeCDF(histogram, cdf, channel.rows * channel.cols);

    // Create a mapping for pixel values based on the CDF
    uchar equalizedMap[L];
    for (int i = 0; i < L; i++) {
        equalizedMap[i] = static_cast<uchar>(255 * cdf[i]);
    }

    // Apply the mapping to the channel
    for (int i = 0; i < channel.rows; i++) {
        for (int j = 0; j < channel.cols; j++) {
            channel.at<uchar>(i, j) = equalizedMap[channel.at<uchar>(i, j)];
        }
    }
}

void processImage(const string &inputPath, const string &csvPath) {
    // Define target resolutions: HD, Full HD, and 4K
    vector<Size> resolutions = { Size(1280, 720), Size(1920, 1080), Size(3840, 2160) };

    // ----- Grayscale Pipeline -----
    Mat grayImage = imread(inputPath, IMREAD_GRAYSCALE);
    if (grayImage.empty()) {
        cerr << "Error: Could not read image from " << inputPath << endl;
        return;
    }
    
    for (const auto &res : resolutions) {
        // Resize the grayscale image to the target resolution
        Mat resizedGray;
        resize(grayImage, resizedGray, res);
        
        // Clone the image for processing so that the original remains intact
        Mat equalizedGray = resizedGray.clone();

        // Start timing for histogram equalization on grayscale image
        auto start = chrono::high_resolution_clock::now();
        performHistogramEqualization(equalizedGray);
        auto end = chrono::high_resolution_clock::now();
        double execTime = chrono::duration<double>(end - start).count();
        
        // Create a resolution string (e.g., "1280x720")
        string resStr = to_string(res.width) + "x" + to_string(res.height);
        
        // Append the resolution, execution time, pipeline type, and number of channels (1 for grayscale) to CSV
        ofstream csvFile(csvPath, ios::app);
        if (csvFile.is_open()) {
            csvFile << resStr << "," << execTime << ",SEQUENTIAL,1\n";
            csvFile.close();
        } else {
            cerr << "Error: Could not open CSV file for writing." << endl;
        }
        
        // Display the original resized grayscale and the equalized image
        imshow("Grayscale Original - " + resStr, resizedGray);
        imshow("Grayscale Equalized - " + resStr, equalizedGray);
        waitKey(0);
        destroyWindow("Grayscale Original - " + resStr);
        destroyWindow("Grayscale Equalized - " + resStr);
    }
    
    // ----- Color Pipeline -----
    Mat colorImage = imread(inputPath, IMREAD_COLOR);
    if (colorImage.empty()) {
        cerr << "Error: Could not read image from " << inputPath << endl;
        return;
    }
    
    for (const auto &res : resolutions) {
        // Resize the color image to the target resolution
        Mat resizedColor;
        resize(colorImage, resizedColor, res);
        
        // Start timing for color image processing
        auto start = chrono::high_resolution_clock::now();
        
        // Convert the resized image to YCrCb color space
        Mat ycrcb;
        cvtColor(resizedColor, ycrcb, COLOR_BGR2YCrCb);
        
        // Split the channels and perform histogram equalization on the Y channel
        vector<Mat> channels;
        split(ycrcb, channels);
        performHistogramEqualization(channels[0]);
        
        // Merge the channels back and convert the image to BGR
        merge(channels, ycrcb);
        Mat equalizedColor;
        cvtColor(ycrcb, equalizedColor, COLOR_YCrCb2BGR);
        
        auto end = chrono::high_resolution_clock::now();
        double execTime = chrono::duration<double, milli>(end - start).count();
        
        // Create a resolution string (e.g., "1920x1080")
        string resStr = to_string(res.width) + "x" + to_string(res.height);
        
        // Append the resolution, execution time, pipeline type, and number of channels (3 for color) to CSV
        ofstream csvFile(csvPath, ios::app);
        if (csvFile.is_open()) {
            csvFile << resStr << "," << execTime << ",SEQUENTIAL,3\n";
            csvFile.close();
        } else {
            cerr << "Error: Could not open CSV file for writing." << endl;
        }
        
        // Display the original resized color image and the equalized result
        imshow("Color Original - " + resStr, resizedColor);
        imshow("Color Equalized - " + resStr, equalizedColor);
        waitKey(0);
        destroyWindow("Color Original - " + resStr);
        destroyWindow("Color Equalized - " + resStr);
    }
}
