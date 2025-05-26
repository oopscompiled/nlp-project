# NLP Project: Emotion Classification from Twitter (X) Dataset

![GitHub](https://img.shields.io/github/license/oopscompiled/nlp-project)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

A natural language processing (NLP) project that classifies emotional personality types from user tweets using supervised machine learning (ML) techniques.

Data sourced from: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

## 📌 Features:

- Emotion classification (e.g., joy, anger, sadness, fear, and love)
- Clean, modular codebase
- Jupyter notebooks for exploration, training, and evaluation
- Data augmentation process
- Custom model and utility functions

## 💡 Skills and Tools Used:

- Natural Language Processing (NLP): tokenization, working with pretrained transformers (DeBERTa)
- Machine Learning: classification using LSTM, GRU, CNN-LSTM with attention mechanisms
- Python 3.8+ and libraries: PyTorch, HuggingFace Transformers, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- Evaluation metrics: accuracy, macro F1-score, precision
- Optimization: Adam and AdamW optimizer, learning rate scheduler with warm-up
- Data analysis and visualization using Jupyter Notebook
- Project management and version control using Git and GitHub

## 📁 Project Structure

```bash
nlp_project/
├── notebooks/              
│   ├── exploratory.ipynb          
│   └── models_training.ipynb       
├── src/                    
│   ├── models.py                   
│   ├── utils.py                    
│   └── data_loader.py           
├── requirements.txt         
├── README.md
├── NLP_report.pdf                      
└── .gitignore                     