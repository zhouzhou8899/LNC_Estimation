# LNC-Estimation
This repository contains the model weights and documentation for the model testing.
 1. Introduction
This repository provides machine learning model files and inference code for predicting leaf nitrogen content (LNC). The models are based on ensemble learning strategies (Random Forest, SVM, XGBoost) and have undergone PSO hyperparameter optimization and feature selection.

 2. Model File Description (Model Files)
The `models/` directory in this repository contains the trained model files (`.pkl`).
Format: Python `joblib` serialization files.
Content: The file contains the complete model object, preprocessing parameters (StandardScaler), feature selection list, and ensemble weights.
Regarding model weights: In this project, the machine learning model weights are packaged together with the model structure in the `.pkl` file. Loading this file restores all model parameter states.
3. Installation
```bash
pip install -r requirements.txt
Python script execution is simple. Run the script and follow the prompts in the dialog boxes.
