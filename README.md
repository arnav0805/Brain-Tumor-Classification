# Brain Tumor Classification using Convolutional Neural Networks (CNN)

## Project Overview

Brain tumors are one of the most critical medical conditions that require early and accurate diagnosis. Manual analysis of MRI scans is time-consuming and highly dependent on expert radiologists. This project aims to automate brain tumor detection and classification using **Convolutional Neural Networks (CNNs)**, a deep learning technique well-suited for image analysis.

The model classifies brain MRI images into multiple tumor categories, helping in faster and more reliable diagnosis support.



## Problem Statement

* Manual MRI analysis is prone to human error and requires specialized expertise.
* Delayed or incorrect diagnosis can severely impact patient outcomes.
* There is a need for an automated, scalable, and accurate system to classify brain tumors from MRI images.



## Objectives

* Build a CNN-based model to classify brain tumor MRI images.
* Achieve high accuracy through effective preprocessing and model training.
* Provide a reproducible pipeline for medical image classification.



##  Dataset

**Source:** Kaggle
**Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Description

The dataset consists of brain MRI images categorized into the following classes:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

The images are already organized into training and testing folders, making it suitable for deep learning workflows.



## Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * TensorFlow – CNN model development
  * OpenCV – Image processing
  * NumPy – Numerical computations
  * Matplotlib – Data visualization
  * Scikit-learn – Model evaluation



## Methodology

1. **Data Preprocessing**

   * Image resizing and normalization
   * Label encoding
   * Train-test split

2. **Model Architecture**

   * Convolutional layers for feature extraction
   * MaxPooling layers for dimensionality reduction
   * Fully connected layers for classification
   * Softmax activation for multi-class output

3. **Model Training**

   * Optimizer: Adam
   * Loss Function: Categorical Crossentropy
   * Evaluation Metric: Accuracy

4. **Model Evaluation**

   * Accuracy and loss curves
   * Confusion matrix
   * Classification report



## Results

* The CNN model achieves high accuracy on test data.
* Demonstrates strong generalization across multiple tumor classes.
* Shows effective feature extraction from MRI images.





##  How to Run the Project

1. Clone the repository:

   ```bash
   git clone <repository-link>
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the project directory.

4. Run the Jupyter Notebook:

   ```bash
   jupyter notebook cnn-brain-tumor.ipynb
   ```



## Use Cases

* Assisting radiologists in tumor diagnosis
* Medical research and academic studies
* AI-driven healthcare applications



##  Challenges Faced

* Handling variations in MRI image quality
* Avoiding overfitting due to limited data
* Selecting optimal CNN architecture

These challenges were addressed using proper preprocessing, regularization techniques, and model tuning.



## Future Enhancements

* Use of transfer learning (VGG16, EfficientNet)
* Model deployment as a web application
* Integration with real-time hospital systems
* Performance improvement using data augmentation


⭐ If you find this project useful, feel free to star the repository!
