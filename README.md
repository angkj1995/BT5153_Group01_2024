# BT5153_Group01_2024

# A Machine Learning Approach to Combat the Spread of Misinformation 
This project applies machine learning techniques to develop a high-accuracy classifier for detecting fake news. It evaluates both traditional NLP approaches (like bag-of-words and TF-IDF) and advanced deep learning methods (including word and sentence embeddings). The project also explores interpretable machine learning methods to increase model transparency.

## Features
This project incorporates several key features to effectively detect and analyze fake news:
- Diverse NLP Techniques: Employs traditional and advanced NLP methods, including Bag-of-Words, TF-IDF, word embeddings, and sentence embeddings.
- Variety of ML Models: Uses multiple machine learning models such as LASSO, CNNs and transformers.
- Comprehensive Dataset: Leverages the LIAR dataset, a broad collection of labeled statements from POLITIFACT.COM, facilitating a detailed analysis of fake news.
- Robust Preprocessing: Implements extensive text preprocessing to ensure data consistency and relevance for model training.
- Detailed Evaluation Metrics: Measures model performance using Precision, Recall, F1-Score, and Accuracy to assess effectiveness across different news categories.
- Explainable AI: Provides model prediction explanations to increase trust and user comprehension, using techniques like Integrated Gradients and feature importance analysis.

## Dataset List
The data is from [LIAR](https://aclanthology.org/P17-2067/) available in 3 files through an 80-10-10 split in the `data01` directory
- test (1,283 rows)
- train (10,269 rows)
- valid (1,284 rows)

## Code Directory
- LIAR_EDA.ipynb: The Exploratory Data Analysis (EDA) notebook provides visualizations and insights into the LIAR dataset's structure and content.
- LIAR_CV_TFIDF_v3.ipynb: Implements CountVectorizer (CV) and Term Frequency-Inverse Document Frequency (TF-IDF) vectorization techniques applied to the LIAR dataset for fake news classification
- LIAR_GloVe_LR_CNN.ipynb: Employs GloVe embeddings for word representation, Logistic Regression (LR) for classification, and a Convolutional Neural Network (CNN) model for capturing local contextual features within the data
- LIAR_DistilBERT.ipynb: Utilizes the DistilBERT transformer model, as an alternative to BERT
- LIAR_sbert.ipynb: Uses Sentence-BERT (sBERT), a modification of the pre-trained BERT model, to produce sentence embeddings that capture semantic similarities for fake news detection tasks
