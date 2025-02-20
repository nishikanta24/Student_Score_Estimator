# Student Score Estimator

## Aim of the Project

The primary goal of this project is to gain a comprehensive understanding of how data science projects are developed and deployed in the industry. By working through this project, I aimed to apply industry-standard code structures and best practices to build a robust machine learning model capable of estimating student scores based on various features.

## Project Structure

```
Student_Score_Estimator/
├── .github/workflows/
│   └── ci-cd-pipeline.yml  # CI/CD workflow for automated testing and deployment
├── .ebextensions/
│   └── [AWS Elastic Beanstalk configuration files]
├── artifacts/
│   └── [Directory for storing trained models and other artifacts]
├── catboost_info/
│   └── [Logs and outputs from CatBoost training]
├── notebook/
│   └── [Jupyter notebooks for exploratory data analysis and experimentation]
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction_pipeline.py
│   └── utils.py
├── templates/
│   └── index.html
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
├── setup.py
└── Dockerfile  # Used for containerization
```

### Key Components
- **Flask for Deployment**: The project uses Flask to deploy the trained model as a web application.
- **CI/CD Pipeline**: A GitHub Actions workflow automates testing, building, and deployment of the model.
- **Machine Learning Algorithms Used**:
  - CatBoost Regressor: Used for training the model due to its efficiency in handling categorical data.
  - Linear Regression: Used as a baseline model for comparison.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nishikanta24/Student_Score_Estimator.git
   cd Student_Score_Estimator
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```
   The application should now be running locally.

## Code Breakdown

### `app.py`
This is the main entry point of the Flask web application.
```python
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text=f'Predicted Score: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
```
- The `/` route renders the homepage (`index.html`).
- The `/predict` route handles form submissions, processes input features, loads the trained model, makes predictions, and returns the result.

### `src/model_training.py`
Handles training the CatBoost model.
```python
from catboost import CatBoostRegressor
import pickle

def train_model(X_train, y_train):
    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=200)
    model.fit(X_train, y_train)
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)
```
- Trains a CatBoost regressor on the provided dataset.
- Saves the trained model to the `artifacts/` directory.


## Results & Insights
- **Model Performance**:
  - CatBoost outperformed linear regression in terms of accuracy.
- **Deployment**:
  - Successfully deployed using Flask.
  - Automated testing and deployment enabled via CI/CD pipeline.

## Conclusion
This project demonstrates the end-to-end development of a machine learning model, from data preprocessing to deployment, following industry best practices. The integration of a CI/CD pipeline and Docker ensures smooth deployment and maintainability.

