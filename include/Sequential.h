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

// Function to process an image, perform histogram equalization, display it, and log execution time
void processImage(const string &inputPath, const string &csvPath) {
    // Start timing
    auto start = chrono::high_resolution_clock::now();

    // Load image
    Mat image = imread(inputPath);
    if (image.empty()) {
        cerr << "Error: Could not read image from " << inputPath << endl;
        return;
    }

    Mat equalizedImage;
    if (image.channels() == 1) {
        // Grayscale image
        equalizedImage = image.clone();
        performHistogramEqualization(equalizedImage);
    } else {
        // For a color image, convert to YCrCb and process the Y channel only
        Mat ycrcb;
        cvtColor(image, ycrcb, COLOR_BGR2YCrCb);

        // Split channels (Y, Cr, Cb)
        vector<Mat> channels;
        split(ycrcb, channels);

        // Equalize the Y channel
        performHistogramEqualization(channels[0]);

        // Merge back and convert to BGR
        merge(channels, ycrcb);
        cvtColor(ycrcb, equalizedImage, COLOR_YCrCb2BGR);
    }

    // Stop timing
    auto end = chrono::high_resolution_clock::now();
    double execTime = chrono::duration<double>(end - start).count();

    // Create resolution string in the format "widthxheight"
    string resolution = to_string(image.cols) + "x" + to_string(image.rows);

    // Append resolution, execution time, and voice type ("SEQUENTIAL") to CSV file
    ofstream csvFile(csvPath, ios::app);
    if (csvFile.is_open()) {
        csvFile << resolution << "," << execTime << ",SEQUENTIAL" << "\n";
        csvFile.close();
    } else {
        cerr << "Error: Could not open CSV file for writing." << endl;
    }

    cout << "Processed image with resolution " << resolution << " in " << execTime << " seconds.\n";

    // Display the original and equalized images
    imshow("Original Image", image);
    imshow("Equalized Image", equalizedImage);

    // Wait for user to press a key
    waitKey(0);
    destroyAllWindows();
}
