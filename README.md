Premier League Match Prediction

This project predicts Premier League match outcomes using historical football data and multiple machine learning models. It was built as a data science project to practice data analysis, feature engineering, and model-based prediction using Python.

The system uses trained models to estimate match outcomes based on team performance statistics.

📌 Project Overview

The project analyzes past Premier League match data and applies different models to predict outcomes. It includes trained models such as Poisson, Ridge, Lasso, and XGBoost, and uses stored model files for prediction.

This project helped me improve my understanding of:

Data preprocessing

Feature engineering

Machine learning models

Model persistence using Pickle

Structuring real-world data science projects

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Pickle

Jupyter Notebook

📂 Project Structure
Premier-League-Prediction/

app.py                     # Main application file  
prem2.ipynb                # Notebook used for experimentation  
models/                    # Saved trained models  
  attack.pkl  
  defense.pkl  
  elo_ratings.pkl  
  feature_scaler.pkl  
  lasso_model.pkl  
  poisson_goals.pkl  
  poisson_model.pkl  
  ridge_model.pkl  
  xgboost_model.pkl  

results.csv                # Prediction results  
stats.csv                  # Team statistics  
with_goalscorers.csv       # Dataset with goal scorers  

⚙️ How to Run the Project

Clone the repository:

git clone https://github.com/MOTORCAR-T/premier-league-prediction


Navigate to the project folder:

cd premier-league-prediction


Install required libraries:

pip install pandas numpy scikit-learn xgboost


Run the project:

python app.py
