# Genre Classification Assignment Summary

## Project Dates:
Project start month: August 2022 \
Project end month: September 2022

## Assignment Overview

This assignment was developed as part of a Kaggle competition, aimed at classifying text sequences extracted from English-language books into four distinct genres: horror, science fiction, humor, and crime fiction. The challenge was to build a machine learning model capable of accurately predicting the genre of unseen text sequences, with performance evaluated using the macro-averaged F1 score.

## Data Description

The dataset consisted of sequences of 10 contiguous sentences from various books, each labeled with one of four genre classes. The training and test datasets were sourced from different sets of books to ensure model generalizability. A unique feature of this dataset was the provision of `docids` for the training data, indicating the book each sequence was extracted from, although this was not available for the test data.

**Note**: The dataset associated with this project is stored externally to control access. It is hosted on Google Drive. You can request viewing of the dataset [here](https://drive.google.com/drive/folders/1ycNtfizE7SyEkKF9Z4Y_D_kDnKTTNpP6?usp=drive_link).


### Data Preparation and Preprocessing

Data preprocessing involved several steps to transform raw text into a machine-readable format:
- **Tokenization**: Text sequences were tokenized into words using the NLTK TreebankWordTokenizer.
- **Encoding**: Words were converted into numerical representations using tokenizers from pre-trained models.
- **Class Weight Calculation**: To address class imbalance, weights were computed for each genre, ensuring the model's sensitivity towards less represented classes.

## Methodology

### Initial Model Exploration

The project began with an exploration of traditional NLP models, including GRU and LSTM neural networks. Despite their proven capabilities in sequence modeling tasks, these models did not perform satisfactorily on the validation and test datasets.

### Adoption of Pre-trained Transformers

Given the limitations of GRU and LSTM models in handling long sequences and the small dataset size, the focus shifted towards leveraging pre-trained transformers. The `distilbert-base-uncased` model, known for its efficiency and performance, was chosen. The model was fine-tuned specifically for this genre classification task using Hugging Face's Transformers library.

#### Fine-tuning Process

Fine-tuning involved the following key steps:
- **Model Selection**: `DistilBertForSequenceClassification` was chosen for its suitability for classification tasks.
- **Training-Validation Split**: The dataset was split into 70% training and 30% validation sets to balance between learning and validation performance.
- **Hyperparameter Optimization**: Key hyperparameters, including learning rate and batch size, were tuned based on validation set performance.

## Challenges and Solutions

### Data Imbalance

The dataset exhibited class imbalance, which could bias the model towards more frequent genres. This was mitigated by calculating class weights during the model training process.

### Long Sequence Handling

Traditional models like GRU and LSTM struggled with the long sequences in the dataset. The transformer model, with its attention mechanism, provided a more effective solution by capturing long-range dependencies in the text.

### Limited Dataset Size

The relatively small size of the dataset posed a challenge for training complex models from scratch. Leveraging a pre-trained model and fine-tuning it on the task-specific dataset proved to be an effective strategy.

## Results

The fine-tuned DistilBERT model demonstrated superior performance, achieving an F1 score of over 0.8 on the validation set. Its performance on the Kaggle test set, with a score of 0.71217, was significantly better than the initial models, underscoring the effectiveness of transformer-based models for this task.

## Conclusion and Future Work

The project highlighted the potential of pre-trained transformer models in tackling NLP classification tasks, especially in scenarios with limited training data and long text sequences. Future work could explore the use of larger or more complex transformer models, ensemble methods, and more extensive hyperparameter tuning to further improve performance.

## Key Learnings and Skills Demonstrated

- Proficiency in data preprocessing and representation for NLP tasks.
- Effective use of pre-trained transformer models for text classification.
- Strategies for handling data imbalance and long sequence data.
- Hyperparameter tuning and model validation techniques.

## Technologies and Tools Used

- **Programming Language**: Python
- **Libraries and Frameworks**: PyTorch, Hugging Face Transformers, NLTK, Sklearn
- **Tools**: Jupyter Notebook, Kaggle platform


## Conclusion
This detailed report reflects the depth of analysis, experimentation, and problem-solving applied in this project, showcasing advanced machine learning techniques and a strategic approach to overcoming challenges in NLP tasks.

<br/><br/>

## **Disclaimer Regarding Dataset:**

The dataset included in this repository is provided solely for educational and portfolio demonstration purposes. It is the property of **Australian National University (ANU)** and may not be redistributed, cloned, or used for any commercial purposes without explicit permission from **Australian National University (ANU)**. 

Users are advised to handle the dataset in compliance with all applicable laws and regulations governing data usage and privacy. By accessing or using the dataset, you agree to abide by these terms and conditions.
