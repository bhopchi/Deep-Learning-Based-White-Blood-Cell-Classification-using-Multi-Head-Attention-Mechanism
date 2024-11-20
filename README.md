# Deep-Learning-Based-White-Blood-Cell-Classification-using-Multi-Head-Attention-Mechanism
"Classifying white blood cells using deep learning with a multi-head attention mechanism for improved accuracy and efficiency."


Detailed Description for GitHub Repository: White Blood Cell Classification Using Deep Learning


---

Project Title**
Deep-Learning-Based-White-Blood-Cell-Classification-using-Multi-Head-Attention-Mechanism**


---

Project Description

This project presents a robust machine learning pipeline for classifying different types of white blood cells (WBCs) using deep learning models enhanced with attention mechanisms. The goal is to assist in automating diagnostic processes in hematology by accurately identifying WBC types based on microscopic images. The project implements state-of-the-art models like Xception and MobileNet, integrated with Multihead Attention layers, to improve feature extraction and classification accuracy.


---

Key Features

1. Dataset

Utilized a Kaggle dataset containing segmented peripheral blood cell images.

Eight classes of WBCs: basophil, eosinophil, erythroblast, lymphocyte, immunoglobulin (ig), monocyte, neutrophil, and platelet.



2. Data Preprocessing

Organized images by their respective labels.

Verified data integrity with checks for null values, duplicates, and imbalanced classes.

Visualized data distribution using count plots and pie charts.



3. Models Implemented

Xception with Multihead Attention:

Modified Xception architecture with spatial attention mechanisms.

Added Gaussian noise and dropout layers for better regularization.


MobileNet with Multihead Attention:

Lightweight architecture ideal for edge devices.

Incorporated spatial attention layers to enhance feature importance.




4. Training and Validation

Split dataset into training (80%), validation (10%), and test (10%) sets.

Rescaled image pixel values for normalized inputs.

Trained models using TensorFlow with Adam optimizer and sparse categorical cross-entropy loss.

Integrated early stopping to prevent overfitting.



5. Performance Metrics

Evaluated models based on accuracy, precision, recall, F1-score, and confusion matrix.

Comparative analysis of Xception and MobileNet F1-scores across classes.

Achieved high test accuracy of 99% with MobileNet.



6. Visual Outputs

Classification metrics visualized through graphs.

Confusion matrices plotted to depict prediction quality.

Comparative bar charts for model F1-scores.





---

Requirements

Python 3.8+

TensorFlow 2.5+

OpenCV

Matplotlib, Seaborn

Pandas, NumPy

Scikit-learn



---

How to Run

1. Clone this repository.

git clone <repository_url>
cd <repository_directory>


2. Install required packages.

pip install -r requirements.txt


3. Prepare the dataset.

Place images in the specified folder structure (original_images/).



4. Train and evaluate models.

Run the main training script:

python train_model.py



5. Visualize results.

Plots and metrics will be saved in the outputs/ directory.





---

Results

Xception Model:

F1-scores ranged from 0.82 to 1.00 across classes.

Highlighted strengths in identifying basophils and lymphocytes.


MobileNet Model:

Improved performance with F1-scores consistently at or near 1.00.

Faster inference times, suitable for real-time applications.




---

Future Enhancements

Expand to multiclass-multilabel classification for complex datasets.

Integrate this system into a real-time diagnostic application.

Experiment with hybrid models combining attention and transformers.



---

License

This project is licensed under the MIT License.


---

Acknowledgments

Kaggle for providing the dataset.

TensorFlow for seamless model implementation.

The academic community for research inspiration.



---
