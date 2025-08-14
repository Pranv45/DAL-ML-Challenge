#!/usr/bin/env python3
"""
Enhanced Medical Chart Embedding Multi-label Classification
Classical ML improvements over the original neural network approach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, classification_report, multilabel_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedMedicalClassifier:
    """Enhanced multi-label classifier for medical chart embeddings"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_thresholds = {}
        self.feature_selector = None
        self.pca_reducer = None
        
    def load_and_preprocess_data(self, data_dir="data"):
        """Load and preprocess the medical chart data"""
        print("Loading and preprocessing data...")
        
        # Load embeddings
        try:
            embeddings_1 = np.load(f"{data_dir}/embeddings_1.npy")
            embeddings_2 = np.load(f"{data_dir}/embeddings_2.npy")
            print(f"Loaded embeddings: {embeddings_1.shape}, {embeddings_2.shape}")
        except FileNotFoundError:
            print("Warning: Could not find embedding files. Using synthetic data for demonstration.")
            # Create synthetic data for demonstration
            embeddings_1 = np.random.randn(10000, 1024)
            embeddings_2 = np.random.randn(10000, 1024)
        
        # Load labels
        try:
            labels_1 = pd.read_csv(f"{data_dir}/icd_codes_1.txt", header=None)
            labels_2 = pd.read_csv(f"{data_dir}/icd_codes_2.txt", header=None)
            print(f"Loaded labels: {labels_1.shape}, {labels_2.shape}")
        except FileNotFoundError:
            print("Warning: Could not find label files. Using synthetic data for demonstration.")
            # Create synthetic ICD codes for demonstration
            icd_codes = [f"ICD{i:03d}" for i in range(100)]
            labels_1 = pd.DataFrame([';'.join(np.random.choice(icd_codes, np.random.randint(1, 5))) 
                                   for _ in range(len(embeddings_1))])
            labels_2 = pd.DataFrame([';'.join(np.random.choice(icd_codes, np.random.randint(1, 5))) 
                                   for _ in range(len(embeddings_2))])
        
        # Concatenate data
        self.all_embeddings = np.concatenate((embeddings_1, embeddings_2), axis=0)
        all_labels = pd.concat((labels_1[0], labels_2[0]), axis=0)
        
        # Process labels
        all_labels_split = all_labels.apply(lambda x: x.split(';') if isinstance(x, str) else [])
        self.mlb.fit(all_labels_split)
        self.multi_hot_labels = self.mlb.transform(all_labels_split)
        
        print(f"Final data shape: {self.all_embeddings.shape}")
        print(f"Label shape: {self.multi_hot_labels.shape}")
        print(f"Number of unique labels: {len(self.mlb.classes_)}")
        
        return self.all_embeddings, self.multi_hot_labels
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data into train/validation/test sets"""
        print("Splitting data...")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.all_embeddings, self.multi_hot_labels, 
            test_size=test_size, random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, random_state=self.random_state
        )
        
        print(f"Train set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def feature_engineering(self, apply_pca=True, pca_components=512, 
                          apply_scaling=True, apply_feature_selection=True, k_best=800):
        """Apply feature engineering techniques"""
        print("Applying feature engineering...")
        
        X_train_processed = self.X_train.copy()
        X_val_processed = self.X_val.copy()
        X_test_processed = self.X_test.copy()
        
        # Standardization
        if apply_scaling:
            print("Applying standardization...")
            X_train_processed = self.scaler.fit_transform(X_train_processed)
            X_val_processed = self.scaler.transform(X_val_processed)
            X_test_processed = self.scaler.transform(X_test_processed)
        
        # PCA for dimensionality reduction
        if apply_pca and pca_components < X_train_processed.shape[1]:
            print(f"Applying PCA to {pca_components} components...")
            self.pca_reducer = PCA(n_components=pca_components, random_state=self.random_state)
            X_train_processed = self.pca_reducer.fit_transform(X_train_processed)
            X_val_processed = self.pca_reducer.transform(X_val_processed)
            X_test_processed = self.pca_reducer.transform(X_test_processed)
            print(f"Explained variance ratio: {self.pca_reducer.explained_variance_ratio_.sum():.3f}")
        
        # Feature selection (if we still have too many features)
        if apply_feature_selection and k_best < X_train_processed.shape[1]:
            print(f"Applying feature selection to {k_best} best features...")
            # Use average target for feature selection
            y_train_avg = self.y_train.mean(axis=1)
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
            X_train_processed = self.feature_selector.fit_transform(X_train_processed, y_train_avg)
            X_val_processed = self.feature_selector.transform(X_val_processed)
            X_test_processed = self.feature_selector.transform(X_test_processed)
        
        # Store processed data
        self.X_train_processed = X_train_processed
        self.X_val_processed = X_val_processed
        self.X_test_processed = X_test_processed
        
        print(f"Final processed feature shape: {X_train_processed.shape}")
        return X_train_processed, X_val_processed, X_test_processed

    def optimize_threshold(self, model_name, y_pred_proba, y_true, metric='f2'):
        """Optimize classification threshold for F2 score"""
        print(f"Optimizing threshold for {model_name}...")
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            if metric == 'f2':
                score = fbeta_score(y_true, y_pred, average='micro', beta=2, zero_division=0)
            else:
                score = fbeta_score(y_true, y_pred, average='micro', beta=1, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.best_thresholds[model_name] = best_threshold
        print(f"Best threshold for {model_name}: {best_threshold:.3f} (Score: {best_score:.4f})")
        return best_threshold

    def train_enhanced_neural_network(self, epochs=50, batch_size=64):
        """Train an enhanced neural network with better architecture"""
        print("Training enhanced neural network...")
        
        input_dim = self.X_train_processed.shape[1]
        output_dim = self.y_train.shape[1]
        
        # Enhanced architecture
        model = Sequential([
            Dense(1024, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='sigmoid')
        ])
        
        # Compile with custom learning rate schedule
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            self.X_train_processed, self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.X_val_processed, self.y_val),
            callbacks=callbacks, verbose=1
        )
        
        # Store model and optimize threshold
        self.models['enhanced_nn'] = model
        
        # Predict and optimize threshold
        y_pred_proba = model.predict(self.X_val_processed)
        self.optimize_threshold('enhanced_nn', y_pred_proba, self.y_val)
        
        return model, history

    def train_classical_models(self):
        """Train various classical ML models"""
        print("Training classical ML models...")
        
        # Use a subset for training classical models if dataset is too large
        max_samples = 50000
        if len(self.X_train_processed) > max_samples:
            print(f"Using subset of {max_samples} samples for classical models...")
            indices = np.random.choice(len(self.X_train_processed), max_samples, replace=False)
            X_train_subset = self.X_train_processed[indices]
            y_train_subset = self.y_train[indices]
        else:
            X_train_subset = self.X_train_processed
            y_train_subset = self.y_train
        
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=20, n_jobs=-1, random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000, n_jobs=-1, random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, n_jobs=-1, random_state=self.random_state
            )
        }
        
        for name, base_model in models_to_train.items():
            print(f"Training {name}...")
            
            # Wrap in MultiOutputClassifier for multi-label
            model = MultiOutputClassifier(base_model, n_jobs=-1)
            model.fit(X_train_subset, y_train_subset)
            
            # Store model
            self.models[name] = model
            
            # Predict probabilities and optimize threshold
            if hasattr(model, 'predict_proba'):
                # For models that support predict_proba
                y_pred_proba = np.array([
                    clf.predict_proba(self.X_val_processed)[:, 1] if hasattr(clf, 'predict_proba')
                    else clf.decision_function(self.X_val_processed)
                    for clf in model.estimators_
                ]).T
            else:
                # For models without predict_proba, use decision_function or predict
                try:
                    y_pred_proba = model.decision_function(self.X_val_processed)
                    # Normalize decision scores to [0,1] range
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                except:
                    # Fallback to binary predictions
                    y_pred_proba = model.predict(self.X_val_processed).astype(float)
            
            self.optimize_threshold(name, y_pred_proba, self.y_val)

    def create_ensemble(self, ensemble_weights=None):
        """Create ensemble predictions from multiple models"""
        print("Creating ensemble predictions...")
        
        if ensemble_weights is None:
            ensemble_weights = {name: 1.0 for name in self.models.keys()}
        
        # Collect predictions from all models
        ensemble_preds = []
        total_weight = sum(ensemble_weights.values())
        
        for name, model in self.models.items():
            if name in ensemble_weights:
                weight = ensemble_weights[name] / total_weight
                
                if 'nn' in name:  # Neural network
                    pred_proba = model.predict(self.X_val_processed)
                elif hasattr(model, 'predict_proba'):
                    pred_proba = np.array([
                        clf.predict_proba(self.X_val_processed)[:, 1] 
                        for clf in model.estimators_
                    ]).T
                else:
                    try:
                        pred_proba = model.decision_function(self.X_val_processed)
                        pred_proba = (pred_proba - pred_proba.min()) / (pred_proba.max() - pred_proba.min())
                    except:
                        pred_proba = model.predict(self.X_val_processed).astype(float)
                
                ensemble_preds.append(weight * pred_proba)
        
        # Average ensemble predictions
        ensemble_pred_proba = np.sum(ensemble_preds, axis=0)
        
        # Optimize threshold for ensemble
        self.optimize_threshold('ensemble', ensemble_pred_proba, self.y_val)
        
        return ensemble_pred_proba

    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name.upper()}...")
            
            # Get predictions
            if 'nn' in name:
                pred_proba = model.predict(self.X_val_processed)
            elif hasattr(model, 'predict_proba'):
                pred_proba = np.array([
                    clf.predict_proba(self.X_val_processed)[:, 1] 
                    for clf in model.estimators_
                ]).T
            else:
                try:
                    pred_proba = model.decision_function(self.X_val_processed)
                    pred_proba = (pred_proba - pred_proba.min()) / (pred_proba.max() - pred_proba.min())
                except:
                    pred_proba = model.predict(self.X_val_processed).astype(float)
            
            # Apply optimized threshold
            threshold = self.best_thresholds.get(name, 0.5)
            pred_binary = (pred_proba > threshold).astype(int)
            
            # Calculate metrics
            f2_score = fbeta_score(self.y_val, pred_binary, average='micro', beta=2, zero_division=0)
            f1_score = fbeta_score(self.y_val, pred_binary, average='micro', beta=1, zero_division=0)
            
            results[name] = {
                'F2_Score': f2_score,
                'F1_Score': f1_score,
                'Threshold': threshold
            }
            
            print(f"F2 Score: {f2_score:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print(f"Optimal Threshold: {threshold:.3f}")
        
        # Evaluate ensemble if multiple models exist
        if len(self.models) > 1:
            print(f"\nEvaluating ENSEMBLE...")
            ensemble_pred_proba = self.create_ensemble()
            threshold = self.best_thresholds.get('ensemble', 0.5)
            pred_binary = (ensemble_pred_proba > threshold).astype(int)
            
            f2_score = fbeta_score(self.y_val, pred_binary, average='micro', beta=2, zero_division=0)
            f1_score = fbeta_score(self.y_val, pred_binary, average='micro', beta=1, zero_division=0)
            
            results['ensemble'] = {
                'F2_Score': f2_score,
                'F1_Score': f1_score,
                'Threshold': threshold
            }
            
            print(f"F2 Score: {f2_score:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print(f"Optimal Threshold: {threshold:.3f}")
        
        # Summary table
        print("\n" + "="*60)
        print("SUMMARY OF ALL MODELS")
        print("="*60)
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('F2_Score', ascending=False)
        print(results_df.round(4))
        
        return results_df

    def save_models(self, save_dir="models"):
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving models to {save_dir}/...")
        
        # Save sklearn models
        for name, model in self.models.items():
            if 'nn' not in name:
                joblib.dump(model, f"{save_dir}/{name}_model.pkl")
        
        # Save neural network models
        for name, model in self.models.items():
            if 'nn' in name:
                model.save(f"{save_dir}/{name}_model.h5")
        
        # Save preprocessing objects
        joblib.dump(self.mlb, f"{save_dir}/multi_label_binarizer.pkl")
        joblib.dump(self.scaler, f"{save_dir}/scaler.pkl")
        if self.pca_reducer:
            joblib.dump(self.pca_reducer, f"{save_dir}/pca_reducer.pkl")
        if self.feature_selector:
            joblib.dump(self.feature_selector, f"{save_dir}/feature_selector.pkl")
        
        # Save thresholds
        joblib.dump(self.best_thresholds, f"{save_dir}/best_thresholds.pkl")
        
        print("All models saved successfully!")

def main():
    """Main execution function"""
    print("Enhanced Medical Chart Embedding Classification")
    print("=" * 50)
    
    # Initialize classifier
    classifier = EnhancedMedicalClassifier(random_state=42)
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data()
    
    # Feature engineering
    X_train_processed, X_val_processed, X_test_processed = classifier.feature_engineering(
        apply_pca=True, pca_components=512,
        apply_scaling=True, apply_feature_selection=True, k_best=400
    )
    
    # Train models
    print("\nTraining models...")
    
    # Train enhanced neural network
    nn_model, history = classifier.train_enhanced_neural_network(epochs=30, batch_size=64)
    
    # Train classical ML models
    classifier.train_classical_models()
    
    # Evaluate all models
    results = classifier.evaluate_all_models()
    
    # Save models
    classifier.save_models()
    
    print("\nTraining and evaluation completed!")
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()