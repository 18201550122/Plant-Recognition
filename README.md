# Plant Recognition using Deep Learning
This project compares the performance of different convolutional neural networks (AlexNet, GoogleNet, ResNet) for plant image recognition. Multiple image enhancement and edge detection methods are applied to improve model accuracy and analyze their impact on classification performance.

# 📁 Project Files
Core Neural Network Models:
Alexnet.py - Implementation of AlexNet architecture.

Googlenet.py - Implementation of GoogleNet (Inception) architecture.

ResNet.py (not provided but mentioned in the paper) - Implementation of ResNet variants (34, 50, 101).

# Preprocessing & Augmentation:
unify_resolution.py - Resizes all images to a uniform resolution (e.g., 1024×768).

# Image Processing:
reinforce.py - Performs data augmentation (cropping, rotation, flipping) to expand the dataset.

edge_detection.py - Applies edge detection algorithms (Sobel, Prewitt, Canny) to images.

# Additional Notes:
The paper also mentions experiments with Laplace, LOG, Gamma, and SSR image enhancement methods, though these may be integrated within the training pipeline or other scripts.
