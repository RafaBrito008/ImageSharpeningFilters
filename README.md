# README for Python Image Processing Application

## Overview
This Python application provides an image processing platform built with Tkinter and OpenCV. It offers various image smoothing and sharpening filters, including Gaussian, median, Laplacian, Sobel, Prewitt, and Roberts filters. Users can load images, apply different filters, and view the results within a user-friendly interface.

## Features
- Load images for processing.
- Apply different smoothing filters: Average, Median, and Gaussian.
- Apply sharpening filters: Laplacian, Sobel, Prewitt, and Roberts.
- Display original and processed images in a graphical user interface.

![sharpening_filters](https://github.com/RafaBrito008/ImageSharpeningFilters/assets/94416107/8cff8e63-4a1e-4c8a-bd08-dc8a5012e27f)


## Requirements
- Python
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Tkinter
- SciPy (for FFT)

## Installation
Ensure you have Python installed along with the required libraries: OpenCV, NumPy, Matplotlib, Tkinter, and SciPy. Most of these can be installed via pip if not already present in your Python environment.

## Usage
Run the script to open the GUI. Use the "Load Image" button to choose an image file, and then apply different filters to see their effects. Processed images are displayed alongside the original image.

## Code Structure
- `SmoothingFilters`: Static methods for applying average, median, and Gaussian filters.
- `SharpeningFilters`: Static methods for applying convolution, Laplacian, Sobel, Prewitt, and Roberts filters.
- `ImageProcessorApp`: Main class to handle the GUI and image processing operations.

## Note
This application is intended for educational and experimental use. Performance may vary depending on the image size and the computing resources available.
