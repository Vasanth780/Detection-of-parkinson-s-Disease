# Detection-of-parkinson-s-Disease
This project focuses on detecting Parkinson's disease using machine learning. By analyzing biomedical voice measurements, such as vocal frequency, amplitude, jitter, and shimmer, the model predicts whether an individual has the disease. The goal is to develop an accurate classification system that aids in early diagnosis and medical research.

Dataset

File: parkinsons_data.csv

Features: The dataset includes multiple attributes such as vocal frequency measurements, amplitude, jitter, shimmer, and other relevant biomedical features.

Target Variable: The presence of Parkinson's disease (binary classification: 1 for presence, 0 for absence).

Installation

To run this project, you need Python and the following libraries:

pip install numpy pandas scikit-learn matplotlib seaborn

Usage

Load the dataset:

import pandas as pd
data = pd.read_csv('parkinsons_data.csv')

Preprocess the data:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('target', axis=1))
y = data['target']

Train a model:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

Evaluate the model:

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

Results

The model's accuracy, precision, recall, and F1-score will be evaluated to determine its effectiveness.

Contributing

If you wish to contribute, feel free to submit issues or pull requests.
