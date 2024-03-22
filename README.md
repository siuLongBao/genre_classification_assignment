# Genre Classification Assignment Summary

## Overview
This project aimed to classify English language book excerpts into four genres: horror, science fiction, humor, and crime fiction, using machine learning. The challenge was part of a Kaggle competition, where the performance was evaluated based on the macro-averaged F1 score. The task involved developing a classification model that accurately identifies the genre of a given text sequence.

## Methodology

### Data Preparation
The dataset comprised text sequences from different books, each labeled with its corresponding genre. The training and test datasets were distinct, ensuring the model's generalizability. The text data was tokenized and encoded using pre-trained models to convert text into a format suitable for machine learning models.

**Note**: The dataset associated with this project is stored externally to control access. It is hosted on Google Drive. You can request viewing of the dataset [here](https://drive.google.com/drive/folders/1ycNtfizE7SyEkKF9Z4Y_D_kDnKTTNpP6?usp=drive_link).

### Model Exploration
Initially, I experimented with GRU and LSTM models to understand their performance on the dataset. However, these models did not yield satisfactory results on the validation and test sets, prompting a shift in strategy.

### Final Solution
The breakthrough came with the adoption of the pre-trained "distilbert-base-uncased" transformer model, specifically the "DistilBertForSequenceClassification." This model was fine-tuned for the genre classification task using the Trainer class from the Transformers library. The choice of DistilBERT was due to its efficiency and relatively lower resource requirements compared to other BERT variants, without a significant compromise on performance.

### Training and Validation
The model was trained with a 70-30 split between training and validation data. This split was chosen after experimenting with different ratios, as it provided a good balance between learning from the data and validating the model's performance. Class weights were calculated to address data imbalance, improving the model's sensitivity to less represented genres. Hyperparameters were fine-tuned based on validation set performance, focusing on learning rate and batch size adjustments.

## Findings
The fine-tuned DistilBERT model achieved a macro-averaged F1 score of over 0.8 on the validation set and 0.71217 on Kaggle, surpassing the performance of initial GRU and LSTM models. This success highlighted the effectiveness of leveraging pre-trained models for natural language processing tasks, especially when dealing with limited data.

## Conclusion
The project demonstrated the power of transformer-based models in text classification tasks. The experience of fine-tuning a pre-trained model provided valuable insights into modern NLP techniques. Despite challenges with data imbalance and model selection, the final solution achieved commendable results in the Kaggle competition.

## Key Learnings
- The importance of data preprocessing and representation in NLP tasks.
- The advantages of using pre-trained models like DistilBERT for complex classification tasks.
- Strategies for addressing data imbalance through class weighting.
- The effectiveness of iterative experimentation with model architectures and hyperparameters.

## Technologies Used
- Python for programming.
- PyTorch and the Transformers library for model implementation and fine-tuning.
- NLTK for text preprocessing and tokenization.
- Sklearn for metrics calculation and evaluation.

This project showcased my ability to apply advanced machine learning techniques to real-world problems, adapt to challenges, and iterate towards an effective solution. The skills and knowledge gained from this experience are invaluable assets for tackling future data science challenges.

## **Disclaimer Regarding Dataset:**

The dataset included in this repository is provided solely for educational and portfolio demonstration purposes. It is the property of **Australian National University (ANU)** and may not be redistributed, cloned, or used for any commercial purposes without explicit permission from **Australian National University (ANU)**. 

Users are advised to handle the dataset in compliance with all applicable laws and regulations governing data usage and privacy. By accessing or using the dataset, you agree to abide by these terms and conditions.
