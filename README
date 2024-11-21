# Breast Histopathology Image Classification Project

## Table of Contents

- [Introduction](#introduction)
- [Research Question](#research-question)
- [Dataset Overview](#dataset-overview)
- [Data Preparation and Understanding](#data-preparation-and-understanding)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Initial Modeling with Simple ML Algorithms](#initial-modeling-with-simple-ml-algorithms)
  - [Before Balancing the Dataset](#before-balancing-the-dataset)
  - [After Balancing the Dataset (Undersampling)](#after-balancing-the-dataset-undersampling)
- [Interpretation of Results](#interpretation-of-results)
- [Next Steps](#next-steps)
- [Conclusion](#conclusion)
- [Appendix](#appendix)
## Introduction

Breast cancer is one of the most common cancers affecting women worldwide. Invasive Ductal Carcinoma (IDC) is the most prevalent subtype, making accurate detection crucial for patient outcomes. Traditional methods rely heavily on pathologists analyzing histopathology slides, which can be time-consuming and subjective. This project aims to leverage machine learning techniques to automate the classification of breast histopathology images as IDC-positive or IDC-negative.
## Research Question

**Can we develop a CNN model to accurately classify breast histopathology image patches as IDC-positive or IDC-negative?**

While our ultimate goal is to implement a Convolutional Neural Network (CNN) for this task, we began by exploring simpler machine learning algorithms to establish baselines and understand the data better.
## Dataset Overview

- **Source**: The dataset consists of breast cancer histopathology images.
- **Total Images**: 277,524 image patches.
- **Classes**:
  - **IDC Negative (Class 0)**: 198,738 images
  - **IDC Positive (Class 1)**: 78,786 images
- **Image Size**: Originally intended to be 50x50 pixels with 3 color channels (RGB), but some images required resizing.
## Data Preparation and Understanding

### Loading the Data

- **Total images**: 277,524
- **Total labels**: 277,524
- **Class distribution**: `Counter({0: 198738, 1: 78786})`

We ensured all images were loaded correctly and labels were extracted based on the filenames.

### Data Cleaning and Preprocessing

- **Handling Missing Data**: Checked for any NaN values or corrupted images and found none.
- **Resizing Images**: Not all images were of size 50x50x3. We resized any images that didn't match this shape to maintain consistency across the dataset.
- **Normalizing Images**: Scaled pixel values to be between 0 and 1 by dividing by 255.
- **Converting Images to Arrays**: Converted all images into NumPy arrays for processing with machine learning models.

### Splitting the Data

- **Training Set**: 80% of the data (222,019 images)
- **Validation Set**: 10% of the data (27,752 images)
- **Test Set**: 10% of the data (27,753 images)

**Data Shapes**:

- `X shape`: (277524, 50, 50, 3)
- `y shape`: (277524,)
- **Training set shape**: (222019, 50, 50, 3)
- **Validation set shape**: (27752, 50, 50, 3)
- **Test set shape**: (27753, 50, 50, 3)
## Exploratory Data Analysis

We visualized a few sample images from each class to get a sense of the data. The images show varying tissue structures and staining patterns, highlighting the complexity of distinguishing between IDC-positive and IDC-negative patches.
## Initial Modeling with Simple ML Algorithms

Before diving into CNNs, we wanted to establish baseline performance using simpler algorithms:

1. **K-Nearest Neighbors (KNN)**
2. **Naive Bayes**
3. **Logistic Regression** (not included in the results here but considered in our experiments)

### Data Preparation for ML Algorithms

- **Flattening Images**: Converted each 50x50x3 image into a 1D array of length 7,500 (50\*50\*3).
- **Reason**: Traditional ML algorithms require 2D input (samples x features).
### Before Balancing the Dataset

#### KNN Results Before Balancing

**KNN Classification Report (Validation Set)**:

          precision    recall  f1-score   support

       0       0.88      0.82      0.85     19874
       1       0.62      0.72      0.66      7878

    accuracy                       0.79     27752
    macro avg  0.75      0.77      0.76     27752
 weighted avg  0.80      0.79      0.80     27752


#### Naive Bayes Results Before Balancing

**Gaussian Naive Bayes Classification Report (Validation Set)**:

            precision    recall  f1-score   support

       0       0.88      0.75      0.81     19874
       1       0.54      0.73      0.62      7878

     accuracy                        0.75     27752
    macro avg    0.71      0.74      0.72     27752
    weighted avg 0.78      0.75      0.76     27752

### After Balancing the Dataset (Undersampling)

#### Balancing Technique: Random Undersampling

- **Original training set shape**: `Counter({0: 158990, 1: 63029})`
- **Undersampled training set shape**: `Counter({1: 63029, 0: 63029})`
- **Method**: Randomly removed samples from the majority class to match the number of samples in the minority class.

#### KNN Results After Undersampling

**KNN Classification Report after Undersampling (Validation Set)**:

                    precision    recall  f1-score   support

                0       0.91      0.69      0.79     19874
                1       0.52      0.83      0.64      7878

            accuracy                        0.73     27752
           macro avg    0.71      0.76      0.71     27752
        weighted avg    0.80      0.73      0.75     27752


#### Naive Bayes Results After Undersampling

**Gaussian Naive Bayes Classification Report after Undersampling (Validation Set)**:

          precision    recall  f1-score   support

            0       0.88      0.75      0.81     19874
            1       0.54      0.73      0.62      7878

         accuracy                       0.75     27752
       macro avg    0.71      0.74      0.72     27752
    weighted avg    0.78      0.75      0.76     27752

## Interpretation of Results

### Before Balancing

- **KNN**:

  For class 0       0.88(precision)      0.82(recall)      0.85(f1)     
  For class 1.      0.62(precision)      0.72(recall)      0.66(f1)   
  Accuracy is 0.79
  - **Observation**: The model favors the majority class but still performs moderately well on the minority class.

- **Naive Bayes**:
  - **Accuracy**: 75%
  - **Class 1 Recall**: 73%
  - **Observation**: Similar to KNN but with slightly lower overall performance.

### After Balancing

- **KNN**:
- 
  For class 0       0.91(precision)      0.69(recall)      0.79(f1)     
  For class 1.      0.52(precision)      0.83(recall)      0.64(f1)   
  Accuracy is 0.73
  - **Accuracy**: Decreased to 73%
  - **Class 1 Recall**: Improved to 83%
  - **Observation**: The recall for IDC-positive cases improved significantly, indicating better detection of the minority class. However, precision for class 1 decreased, suggesting more false positives.

- **Naive Bayes**:
  - **Results** remained relatively consistent with pre-balancing, indicating that Naive Bayes may not be as sensitive to class imbalance or the undersampling didn't impact its performance as much.

### Trade-offs

- **Loss of Data**: Undersampling reduced the number of samples in the majority class, potentially discarding valuable information.
- **Model Bias**: Post-undersampling, models became less biased towards the majority class, improving detection of IDC-positive cases.
- **Overall Accuracy**: Slight decrease in overall accuracy but improved performance on the minority class, which is crucial in medical diagnoses.

## Next Steps

Given the visual nature of the data and the limitations observed with simple ML algorithms, especially after undersampling, we plan to:

- **Implement Convolutional Neural Networks (CNNs)**:
  - CNNs are designed to capture spatial hierarchies in images, potentially improving classification performance without the need to flatten images.
  - We expect CNNs to handle class imbalance more effectively due to their ability to learn complex patterns.
- **Data Augmentation**:
  - Increase the diversity of the dataset by applying transformations like rotations, flips, and zooms.
  - Helps in preventing overfitting and improving the model's ability to generalize.
- **Experiment with Different Architectures**:
  - Test various CNN architectures and hyperparameters to find the optimal model.
- **Implement Transfer Learning**:
  - Use pre-trained models like VGG16 or ResNet50 to leverage learned features from larger datasets.
## Appendix

### Data Processing Code Snippets

```python
# Example code for resizing images
from PIL import Image

def resize_image(img, size=(50, 50)):
    return img.resize(size)

# Example code for flattening images
X_flat = X.reshape(X.shape[0], -1)

---
```

### 12. Final Note

---

**Note**: This README summarizes the steps and findings of our project up to this point. As we proceed with implementing CNNs, we will update this document with new results and insights.

---

# End of README
