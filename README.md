# Histogram-Equalization

Histogram Equalization is a technique used to enhance the contrast of an image by redistributing its pixel intensity values. The algorithm works by computing the histogram of the image, calculating the cumulative distribution function (CDF), and then mapping the original pixel values to new values based on the CDF. This results in an image with a more uniform distribution of intensities, improving its visual quality. The project implements this algorithm using both sequential and parallel approaches, leveraging CUDA for GPU acceleration.

## Project Structure

This repository contains the following files and directories:

### Root Files
- **CMakeLists.txt**: The CMake configuration file for building the project.
- **execution_times.csv**: A CSV file containing execution time results for different implementations.
- **main.cu**: The main CUDA source file that implements the histogram equalization algorithm.
- **plot.py**: A Python script for visualizing execution times or other results.

### `imgs/`
Contains input and output images used for testing and visualization.

### `include/`
Contains header files for the project:
- **Kernels.cuh**: CUDA kernel definitions for parallel processing.
- **Parallel.cuh**: Header file for parallel implementation functions.
- **Sequential.h**: Header file for sequential implementation functions.


