# NBA Shot Success Prediction

## Overview
This project predicts NBA shot outcomes using advanced machine learning (XGBoost), with engineered features for shot zone, defender proximity, and shot clock timing.

## Key Features
- Feature engineering (shot zone, defender proximity, shot clock, interaction terms)
- Addressed class imbalance with SMOTE
- Hyperparameter tuning with GridSearchCV
- Model evaluation (accuracy, precision, recall, F1, ROC-AUC)
- Visualizations: feature importance, confusion matrix, ROC curve

## Results
- XGBoost outperformed baseline models in accuracy and F1 score
- Shot distance was the most important predictor of shot success

## How to Run
1. Clone the repo  
2. Install requirements:  
   `pip install -r requirements.txt`  
3. Place `shot_logs.csv` in the root directory  
4. Run `project.py`

## License
MIT
