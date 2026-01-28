
# Brain Tumor Classification using Convolutional Neural Networks (CNN)

##  Project Overview

Brain tumors are one of the most critical medical conditions that require early and accurate diagnosis. Manual analysis of MRI scans is time-consuming and highly dependent on expert radiologists. This project automates brain tumor detection and classification using **Convolutional Neural Networks (CNNs)** combined with **EfficientNet**, a powerful pre-trained deep learning model.

The model classifies brain MRI images into multiple tumor categories, enabling faster and more reliable diagnosis support.


## Problem Statement

- Manual MRI analysis is prone to human error and requires specialized expertise.
- Delayed or incorrect diagnosis can severely impact patient outcomes.
- There is a need for an automated, scalable, and accurate system to classify brain tumors from MRI images.



## Objectives

- Build a hybrid deep learning model using **Custom CNN + EfficientNet**.
- Achieve high accuracy through effective preprocessing and transfer learning.
- Provide a reproducible pipeline for medical image classification.



## Dataset

**Source:** Kaggle  
**Dataset Link:**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Dataset Description

The dataset consists of brain MRI images categorized into four classes:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The images are pre-organized into training and testing folders, making them suitable for deep learning workflows.

---

##  Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- TensorFlow– Model development
- EfficientNet – Transfer learning
- OpenCV – Image preprocessing
- NumPy – Numerical computations
- Matplotlib – Visualization
- Scikit-learn – Model evaluation



## Methodology

### 1️Data Preprocessing
- Image resizing and normalization
- Label encoding
- Train-test split

### 2️Model Architecture
- Initial **custom CNN layers** for low-level feature extraction
- **EfficientNet (pre-trained)** stacked on top of the custom CNN
- MaxPooling layers for dimensionality reduction
- Fully connected (Dense) layers for classification
- Softmax activation for multi-class output

This hybrid approach combines the flexibility of a custom CNN with the strong feature representation of EfficientNet.

### Model Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Evaluation Metric: Accuracy
- Transfer learning applied by freezing initial EfficientNet layers and fine-tuning higher layers

### Model Evaluation
- Accuracy and loss curves
- Confusion matrix
- Classification report



##  Results

- The **Custom CNN + EfficientNet** model achieves high accuracy on test data.
- Shows better generalization compared to a standalone CNN.
- EfficientNet improves feature extraction and classification performance across all tumor classes.



## How to Run the Project

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
##  Use Cases

* Assisting radiologists in tumor diagnosis
* Medical research and academic projects
* AI-powered healthcare decision support systems


##  Challenges Faced

* Handling variations in MRI image quality
* Integrating EfficientNet with a custom CNN architecture
* Preventing overfitting during fine-tuning
* Managing computational cost of deep models

These challenges were addressed using proper preprocessing, regularization techniques, and model tuning.

## Future Enhancements

* Further fine-tuning of EfficientNet layers
* Experimenting with other pre-trained models (ResNet, DenseNet)
* Deploying the model as a web or mobile application
* Improving robustness using advanced data augmentation techniques


