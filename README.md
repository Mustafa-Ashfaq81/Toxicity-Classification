# Toxicity Classification

## Overview
This project focuses on automating the detection and classification of toxic behavior in user-generated content, using machine learning models to predict various types of toxicity in Wikipedia comments. The project explores different machine learning approaches to create a robust model that contributes to safer digital communication spaces.

## About the Data
The dataset comprises Wikipedia comments labeled by human raters for toxic behavior, including:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Data Preprocessing
- Removal of stop words, URLs, non-alphanumeric characters, and extra whitespaces using the NLTK library and regex.
- Utilization of `TF-IDF` vectorization and `Embed4all` feature extractor for numerical representation of text data.

## Models Explored
- **Multinomial Naïve Bayes Classifier**
- **Logistic Regression Classifier**
- **Linear Support Vector Classifier**
- **Custom Neural Network**
- **Pretrained Encoder-Transformers (BERT)**

## Key Findings
- The project highlighted the effectiveness of different machine learning models in toxicity classification.
- Advanced models like Artificial Neural Networks and Pre-Trained Encoded Transformers showed promising results, outperforming traditional classification models.
- Class imbalance was addressed using the `LeastSampleClassSampler` technique, improving model performance.

## Technologies Used
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- TensorFlow, PyTorch
- BERT (Bidirectional Encoder Representations from Transformers)

## Installation
Clone this repository and install the required libraries:
```bash
git clone https://github.com/Mustafa-Ashfaq81/Toxicity-Classification.git
```

## Usage
Refer to the Jupyter notebooks and Python scripts in the repository to run the models and analyze the results.
