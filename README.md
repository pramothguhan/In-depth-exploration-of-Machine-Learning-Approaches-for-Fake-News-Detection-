## Fake News Detection :newspaper:

## Overview :memo:

The rapid spread of fake news has become a significant concern with the potential to cause harm to individuals, institutions, and society. This project explores machine learning approaches for detecting fake news by analyzing the content and structure of news articles. The project utilizes a variety of machine learning classifiers, including Logistic Regression, Decision Trees, and Random Forests, as well as advanced deep learning models like LSTMs, to identify patterns indicative of fake news.

## Project Highlights :sparkles:

- **Dataset:** The WELFake dataset, consisting of 72,134 news articles, with 35,028 labeled as real and 37,106 labeled as fake, was used for training and evaluation. The dataset was compiled by merging four popular news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political).
- **Techniques:** The project employed various natural language processing (NLP) techniques, including tokenization, stemming, and stop word removal, for feature extraction from the text of news articles.
- **Models:** Multiple machine learning models were implemented and compared, including Naive Bayes, Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM). Additionally, LSTM models were developed and compared using different embeddings.
- **Evaluation:** The performance of the models was assessed using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.

## Technologies Used :computer:

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![NLTK](https://img.shields.io/badge/NLTK-2F4F4F?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)

## Methodology :gear:

### Data Preprocessing:
- **Tokenization, Stemming, and Stopword Removal:** Applied to clean and standardize the text data for feature extraction.
- **TF-IDF Vectorization:** Used to convert the text data into numerical features for input into machine learning models.

### Model Development:
1. **Naive Bayes:**
   - Implemented using `MultinomialNB` from Scikit-Learn.
   - Pipeline included vectorization and TF-IDF transformation.
   
2. **Logistic Regression:**
   - Implemented using `LogisticRegression` from Scikit-Learn.
   - Evaluated using accuracy metrics on the test data.

3. **Decision Tree:**
   - Implemented using `DecisionTreeClassifier` with the entropy criterion and a max depth of 20.
   - Performed well with high accuracy on test data.

4. **Random Forest:**
   - Implemented using `RandomForestClassifier` with 50 estimators and the entropy criterion.
   - Demonstrated strong performance in classifying fake news.

5. **Support Vector Machine (SVM):**
   - Implemented using `SVC` with a linear kernel.
   - Showed competitive accuracy in detecting fake news.

6. **LSTM Models:**
   - Developed using TensorFlow and Keras to handle sequential data.
   - Compared different embeddings and evaluated their effectiveness in classification.

## Results :chart_with_upwards_trend:

- **Best Performing Model:** The Decision Tree model achieved the highest accuracy among the tested classifiers, making it the most effective model for fake news detection in this project.
- **Model Comparisons:** Detailed comparisons were made across various models, highlighting the strengths and limitations of each approach.

## Project Folder Structure :file_folder:
```
📦 Fake_News_Detection
├── data
├── notebooks
│ └── FakeNewsDetection.ipynb
├── src
│ ├── data_preprocessing.py
│ ├── model_training.py
│ └── evaluation.py
├── results
│ └── model_comparisons.png
└── README.md
```

## Conclusion and Future Work :mag:
The project successfully demonstrated the use of machine learning techniques in detecting fake news, with the Decision Tree model emerging as the most effective. Future work will focus on further refining the models, exploring deep learning architectures, and expanding the dataset to include a more diverse range of news sources.
