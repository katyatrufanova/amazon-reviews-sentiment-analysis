# Amazon Reviews Sentiment Analysis

## 🌟 Project Overview

This repository contains a comprehensive exploration of cutting-edge machine learning models and techniques for sentiment analysis on Amazon product reviews. The project aims to classify reviews into positive, negative, or neutral categories, leveraging state-of-the-art NLP techniques and deep learning models.

### 🎯 Key Features

- Implementation of various ML models including Naive Bayes, Logistic Regression, SVM, biLSTM, and DistilBERT
- Advanced techniques such as ensemble learning, LIME (Local Interpretable Model-agnostic Explanations), SMOTE (Synthetic Minority Over-sampling Technique), and more
- Comparative analysis of model performance using statistical tests (Friedman test)
- Interactive web interface for real-time sentiment prediction
- Extensive documentation including a detailed project report and presentation

## 📊 Models and Techniques

1. **Traditional Machine Learning Models**
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machines (SVM)

2. **Deep Learning Models**
   - Bidirectional LSTM (with and without GloVe embeddings)
   - DistilBERT (SOTA Transformer model)

3. **Advanced Techniques**
   - Ensemble Learning (Voting, Bagging, Boosting)
   - LIME for model interpretability
   - SMOTE for handling imbalanced data
   - Non-negative Matrix Factorization (NMF) for dimensionality reduction
   - Feature engineering and selection

## 📂 Project Organization
```
amazon-reviews-sentiment-analysis/
├── notebooks/
│   ├── 01_NB_LR_SVM_Ensemble_LIME.ipynb
│   ├── 02_NB_LR_SVM_NMF_SMOTE.ipynb
│   ├── 03_NB_LR_SVM_Feature_Engineering.ipynb
│   ├── 04_biLSTM.ipynb
│   ├── 05_DistilBERT.ipynb
│   └── 06_Visualization_Friedman.ipynb
├── docs/
│   ├── Report.pdf
│   └── Presentation.pdf
├── src/
│   ├── sentiment_api.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── script.js
│   └── templates/
│       └── index.html
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip
- virtualenv (optional, but recommended)

### Installation

1. Clone the repository:
```
git clone https://github.com/katyatrufanova/amazon-reviews-sentiment-analysis.git
cd amazon-reviews-sentiment-analysis
```

2. Create and activate a virtual environment (optional):
```
python -m venv myenv
source myenv/bin/activate  # On Windows, use myenv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Download the pre-trained model:
```
wget https://drive.google.com/file/d/19uTB-PISY7nhdQuDYxZG9KW02nBoSMvL/view?usp=drive_link -O src/sentiment_model_weights.h5
```

## 💻 Usage

### Running the Notebooks

1. Upload the [Amazon Reviews dataset](https://www.kaggle.com/datasets/PromptCloudHQ/Amazon-reviews-unlocked-mobile-phones) to your Google Drive in the path: `MyDrive/ML/Amazon_Unlocked_Mobile.csv`

2. Open the notebooks in Google Colab and run them sequentially:
- `01_NB_LR_SVM_Ensemble_LIME.ipynb`
- `02_NB_LR_SVM_NMF_SMOTE.ipynb`
- `03_NB_LR_SVM_Feature_Engineering.ipynb`
- `04_biLSTM.ipynb`
- `05_DistilBERT.ipynb`
- `06_Visualization_Friedman.ipynb`

### Using the Web Interface

1. Navigate to the `src` directory:
```
cd src
```

2. Run the Flask application:
```
python sentiment_api.py
```

3. Open your web browser and navigate to `http://localhost:5000`

4. Enter a product review in the text box and click "Analyze" to get the sentiment prediction

## 📈 Results

The conducted experiments revealed several interesting findings:

- State-of-the-art models (biLSTM and DistilBERT) demonstrated the highest performance
- Simpler models (Naive Bayes, Logistic Regression, SVM) showed robust performance, exceeding initial expectations
- The performance difference between biLSTM models with and without GloVe embeddings was minimal
- Advanced techniques like feature engineering and SMOTE did not significantly improve model performance in this context

For a detailed analysis of the results, please refer to the [full project report](docs/Report.pdf).

## 🛠️ Technologies Used

- Python
- TensorFlow
- PyTorch
- scikit-learn
- NLTK
- Flask
- HTML/CSS/JavaScript
