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

// Convert from RGB to YCbCr; note: OpenCV uses BGR order.
void RGBtoYCbCr(const unsigned char &R, const unsigned char &G, const unsigned char &B,
    unsigned char &Y, unsigned char &Cb, unsigned char &Cr) {
    Y = static_cast<unsigned char>(0.299 * R + 0.587 * G + 0.114 * B);
    Cb = static_cast<unsigned char>(128 - 0.168736 * R - 0.331264 * G + 0.5 * B);
    Cr = static_cast<unsigned char>(128 + 0.5 * R - 0.418688 * G - 0.081312 * B);
}

// Convert from YCbCr to RGB.
void YCbCrtoRGB(const unsigned char &Y, const unsigned char &Cb, const unsigned char &Cr,
    unsigned char &R, unsigned char &G, unsigned char &B) {
    R = static_cast<unsigned char>(min(255.0, max(0.0, Y + 1.402 * (Cr - 128))));
    G = static_cast<unsigned char>(min(255.0, max(0.0, Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
    B = static_cast<unsigned char>(min(255.0, max(0.0, Y + 1.772 * (Cb - 128))));
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
        double execTime = chrono::duration<double, milli>(end - start).count();
        
        // Create a resolution string (e.g., "1280x720")
        string resStr = to_string(res.width) + "x" + to_string(res.height);
        
        // Append the resolution, execution time, pipeline type, and number of channels (1 for grayscale) to CSV
        {
            ifstream checkFile(csvPath);
            bool isEmpty = checkFile.peek() == ifstream::traits_type::eof();
            checkFile.close();

            ofstream csvFile(csvPath, ios::app);
            if (csvFile.is_open()) {
                if (isEmpty) {
                    csvFile << "Resolution,Time,Type,Channels\n";
                }
                csvFile << resStr << "," << execTime << ",SEQUENTIAL,1\n";
                csvFile.close();
            } else {
                cerr << "Error: Could not open CSV file for writing." << endl;
            }
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
        auto start = chrono::high_resolution_clock::now();

        // Prepare YCbCr channels
        Mat yChannel(resizedColor.size(), CV_8UC1);
        Mat cbChannel(resizedColor.size(), CV_8UC1);
        Mat crChannel(resizedColor.size(), CV_8UC1);

        // Convert each pixel from RGB to YCbCr
        for (int i = 0; i < resizedColor.rows; i++) {
            for (int j = 0; j < resizedColor.cols; j++) {
                Vec3b pixel = resizedColor.at<Vec3b>(i, j);
                unsigned char Y, Cb, Cr;
                // OpenCV uses BGR order
                RGBtoYCbCr(pixel[2], pixel[1], pixel[0], Y, Cb, Cr);
                yChannel.at<uchar>(i, j) = Y;
                cbChannel.at<uchar>(i, j) = Cb;
                crChannel.at<uchar>(i, j) = Cr;
            }
        }

        // Perform histogram equalization on the Y channel
        performHistogramEqualization(yChannel);

        // Convert YCbCr back to RGB
        Mat equalizedColor(resizedColor.size(), resizedColor.type());
        for (int i = 0; i < resizedColor.rows; i++) {
            for (int j = 0; j < resizedColor.cols; j++) {
                unsigned char R, G, B;
                YCbCrtoRGB(yChannel.at<uchar>(i, j), cbChannel.at<uchar>(i, j), crChannel.at<uchar>(i, j), R, G, B);
                // OpenCV uses BGR order
                equalizedColor.at<Vec3b>(i, j) = Vec3b(B, G, R);
            }
        }
        auto end = chrono::high_resolution_clock::now();
        double execTime = chrono::duration<double, milli>(end - start).count();

        // Create a resolution string (e.g., "1920x1080")
        string resStr = to_string(res.width) + "x" + to_string(res.height);
        
        // Append the resolution, execution time, pipeline type, and number of channels (3 for color) to CSV
        {
            ifstream checkFile(csvPath);
            bool isEmpty = checkFile.peek() == ifstream::traits_type::eof();
            checkFile.close();

            ofstream csvFile(csvPath, ios::app);
            if (csvFile.is_open()) {
                if (isEmpty) {
                    csvFile << "Resolution,Time,Type,Channels\n";
                }
                csvFile << resStr << "," << execTime << ",SEQUENTIAL,3\n";
                csvFile.close();
            } else {
                cerr << "Error: Could not open CSV file for writing." << endl;
            }
        }
        
        // Display the original resized color image and the equalized result
        imshow("Color Original - " + resStr, resizedColor);
        imshow("Color Equalized - " + resStr, equalizedColor);
        waitKey(0);
        destroyWindow("Color Original - " + resStr);
        destroyWindow("Color Equalized - " + resStr);
    }
}

