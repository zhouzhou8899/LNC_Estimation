import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA

warnings.filterwarnings('ignore')

class CustomPCA:
    def __init__(self, n_components=0.85, n_feat=10, method="pca"):
        self.n_components = n_components
        self.n_feat = n_feat
        self.method = method
        self.pca = None
        self.explained_variance_ratio_ = None
        self.loadings_ = None
        self.components_ = None
        self.X = None
        self.selected_features = None

    def fit_transform(self, X, row_labels=None, col_labels=None):
        pass

    def transform(self, X, row_labels=None, col_labels=None):
        pass

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure CustomPCA class is defined. Error: {e}")

def predict(model_package, input_data_path, output_path='prediction_result.csv'):
    print(f"Loading data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    print(f"Data loaded: {df.shape}")
    
    feature_key = 'selected_features' if 'selected_features' in model_package else 'selected_features_final'
    required_features = model_package[feature_key]
    
    missing_cols = set(required_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input data missing required features: {missing_cols}")
    
    X = df[required_features].values
    
    scaler_X = model_package.get('scaler_X')
    if scaler_X is None:
        raise ValueError("Model package missing scaler_X")
        
    X_scaled = scaler_X.transform(X)
    
    rf_model = model_package.get('rf_model')
    svm_model = model_package.get('svm_model')
    xgb_model = model_package.get('xgb_model')
    
    if not all([rf_model, svm_model, xgb_model]):
        raise ValueError("Model package missing base models (rf_model, svm_model, xgb_model)")

    pred_rf = rf_model.predict(X_scaled)
    pred_svm = svm_model.predict(X_scaled)
    pred_xgb = xgb_model.predict(X_scaled)
    
    weights_key = 'weights' if 'weights' in model_package else 'meta_model_weights'
    weights = model_package[weights_key]
    
    ensemble_pred_scaled = (pred_rf * weights[0] + 
                            pred_svm * weights[1] + 
                            pred_xgb * weights[2])
    
    scaler_y = model_package.get('scaler_y')
    if scaler_y is None:
        raise ValueError("Model package missing scaler_y")
        
    final_pred = scaler_y.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).flatten()
    
    result_df = df.copy()
    result_df['Predicted_LNC'] = final_pred
    result_df.to_csv(output_path, index=False)
    print(f"Prediction completed. Results saved to {output_path}")
    return final_pred

if __name__ == "__main__":
    # ================= Configuration Area (Reviewers may edit this section) =================
    # Select the model file to be tested
    # Option 1: baoding ensemble model
    # model_path = 'models/baoding_ensemble_model.pkl' 
    
    # Option 2:transfer learning model: Wei county_A  (A:JND36/SK126)
    model_path = 'model_weights/lnc_prediction_unified_transfer_Wei_county_JND36.pkl'
    input_path = 'sample_data/test_input.csv'
    # =============================================================
    
    try:
        package = load_model(model_path)
        predict(package, input_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
