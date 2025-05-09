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

## Future Plans

This project is the foundation for a larger idea: creating a personalized app where users can input their own basketball shot data, including details like shot location, defender distance, player height/weight, and shot clock timing and receive feedback based on comparisons to real NBA player data.

Planned features:
- A web app interface where users can log their own shooting sessions
- Personalized performance feedback based on similarity to NBA players’ shot patterns
- Visual comparisons of shot accuracy by zone, timing, and defensive pressure
- Coaching-style recommendations on where to improve based on model insights
- Potential integration of computer vision or tracking data for automated shot logging

This would turn the current machine learning model into an interactive tool for players, coaches, and fans looking to analyze and improve their shooting performance through data.



## License
MIT
