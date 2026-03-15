#!/usr/bin/env python3
"""
Cirrhosis Patient Survival Prediction Model
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import typer
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

app = typer.Typer()


logger.remove()
logger.add(
    "data/log_file.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(sys.stdout, level="INFO")

class My_Classifier_Model:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        
    def _preprocess_data(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess the input data"""
        logger.info(f"Preprocessing data (is_train={is_train})")
        data = df.copy()
        
        if 'id' in data.columns:
            self.ids = data['id'] if not is_train else None
            data = data.drop('id', axis=1)
        
        if is_train and 'Status' in data.columns:
            y = data['Status'].copy()
            data = data.drop('Status', axis=1)
            logger.info(f"Target distribution:\n{y.value_counts()}")
        else:
            y = None
        
        for col in self.categorical_cols:
            if col in data.columns:
                if is_train:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        data[col] = data[col].astype(str)
                        mapping = {label: i for i, label in enumerate(le.classes_)}
                        data[col] = data[col].map(mapping).fillna(-1).astype(int)
        
        logger.info(f"Preprocessing complete. Shape: {data.shape}")
        return data, y
    
    def train(self, dataset_filename: str) -> None:
        """Train the model"""
        logger.info(f"Starting training with dataset: {dataset_filename}")
        
        Path("./model").mkdir(exist_ok=True)
        Path("./data").mkdir(exist_ok=True)
        
        df = pd.read_csv(dataset_filename)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        X, y = self._preprocess_data(df, is_train=True)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
       
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,
            border_count=128,
            bagging_temperature=0.5,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            verbose=100,
            random_seed=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )
        
        val_pred = self.model.predict_proba(X_val)
        val_loss = log_loss(y_val, val_pred)
        logger.info(f"Validation Log Loss: {val_loss:.4f}")
        
        artifacts = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'categorical_cols': self.categorical_cols
        }
        
        with open("./model/model.pkl", 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.success(f"Model saved to ./model/model.pkl")
        logger.success(f"Training completed! Validation Log Loss: {val_loss:.4f}")
    
    def predict(self, dataset_filename: str) -> None:
        """Make predictions"""
        logger.info(f"Starting prediction with dataset: {dataset_filename}")
        
        if not os.path.exists("./model/model.pkl"):
            logger.error("Model not found. Please train first.")
            raise FileNotFoundError("Model not found. Please train first.")
        
        with open("./model/model.pkl", 'rb') as f:
            artifacts = pickle.load(f)
        
        self.model = artifacts['model']
        self.label_encoders = artifacts['label_encoders']
        
        df = pd.read_csv(dataset_filename)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        self.ids = df['id'] if 'id' in df.columns else None
        X, _ = self._preprocess_data(df, is_train=False)
        
        predictions = self.model.predict_proba(X)
        
        if self.ids is not None:
            result = pd.DataFrame({
                'id': self.ids,
                'Status_C': predictions[:, 0],
                'Status_CL': predictions[:, 1],
                'Status_D': predictions[:, 2]
            })
        else:
            result = pd.DataFrame(
                predictions,
                columns=['Status_C', 'Status_CL', 'Status_D']
            )
        
        result.to_csv("./data/results.csv", index=False)
        logger.success(f"Predictions saved to ./data/results.csv")
        
        logger.info(f"Prediction statistics:")
        logger.info(f"Status_C mean: {predictions[:, 0].mean():.4f}")
        logger.info(f"Status_CL mean: {predictions[:, 1].mean():.4f}")
        logger.info(f"Status_D mean: {predictions[:, 2].mean():.4f}")

@app.command()
def train(dataset: str = typer.Argument(..., help="Path to training dataset")):
    """Train the model"""
    model = My_Classifier_Model()
    model.train(dataset)

@app.command()
def predict(dataset: str = typer.Argument(..., help="Path to evaluation dataset")):
    """Make predictions"""
    model = My_Classifier_Model()
    model.predict(dataset)

if __name__ == "__main__":
    app()
