#!/usr/bin/env python3
"""
Comprehensive Enhanced Medical Chart Embedding Multi-label Classification
Integrates all classical ML improvements: threshold optimization, class imbalance handling,
label correlation modeling, ensemble methods, and feature engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from class_imbalance_handler import MultiLabelImbalanceHandler
from label_correlation_models import ClassifierChainModel, EnsembleClassifierChains

class ComprehensiveEnhancedClassifier:
    """Comprehensive enhanced multi-label classifier with all improvements"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_thresholds = {}
        self.feature_selector = None
        self.pca_reducer = None
        self.imbalance_handler = MultiLabelImbalanceHandler(random_state)

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

        # Analyze class imbalance
        stats, report = self.imbalance_handler.analyze_label_distribution(
            self.multi_hot_labels, self.mlb.classes_
        )

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

    def apply_feature_engineering(self, apply_pca=True, pca_components=512,
                                apply_scaling=True, apply_feature_selection=True, k_best=400):
        """Apply comprehensive feature engineering"""
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

        # Feature selection
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

    def handle_class_imbalance(self, apply_sampling=True, apply_class_weights=True):
        """Handle class imbalance using multiple techniques"""
        print("Handling class imbalance...")

        # Apply sampling techniques
        if apply_sampling:
            print("Applying label-specific oversampling...")
            self.X_train_processed, self.y_train = self.imbalance_handler.apply_multilabel_sampling(
                self.X_train_processed, self.y_train, method='label_specific_oversample'
            )

        # Compute class weights
        if apply_class_weights:
            print("Computing class weights...")
            self.class_weights = self.imbalance_handler.compute_class_weights(
                self.y_train, method='balanced'
            )

        return self.X_train_processed, self.y_train

    def optimize_threshold(self, model_name, y_pred_proba, y_true, metric='f2'):
        """Optimize classification threshold for F2 score"""
        print(f"Optimizing threshold for {model_name}...")

        # Reduce threshold optimization range for faster execution
        thresholds = np.arange(0.2, 0.8, 0.1)
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

    def train_enhanced_neural_network(self, epochs=12, batch_size=64):
        """Train an enhanced neural network with imbalance handling"""
        print("Training enhanced neural network...")

        # Check if checkpoint exists
        checkpoint_path = 'checkpoints/enhanced_nn_best.h5'
        if os.path.exists(checkpoint_path):
            print(f"Found existing checkpoint at {checkpoint_path}")
            try:
                model = tf.keras.models.load_model(checkpoint_path)
                print("Successfully loaded model from checkpoint!")
                self.models['enhanced_nn'] = model

                # Still optimize threshold with loaded model
                y_pred_proba = model.predict(self.X_val_processed)
                self.optimize_threshold('enhanced_nn', y_pred_proba, self.y_val)
                return model, None
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Training from scratch...")

        input_dim = self.X_train_processed.shape[1]
        output_dim = self.y_train.shape[1]

        # Enhanced architecture with batch normalization
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

        # Use weighted loss for imbalance handling
        if hasattr(self, 'class_weights'):
            custom_loss = self.imbalance_handler.create_weighted_loss(
                self.class_weights, loss_type='weighted_binary_crossentropy'
            )
            model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['accuracy'])
        else:
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Callbacks for better training
        from tensorflow.keras.callbacks import ModelCheckpoint
        os.makedirs("checkpoints", exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                filepath='checkpoints/enhanced_nn_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
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

    def train_classical_ensemble(self):
        """Train ensemble of classical ML models"""
        print("Training classical ML ensemble...")

        # Use subset for classical models if dataset is too large
        max_samples = 15000
        if len(self.X_train_processed) > max_samples:
            print(f"Using subset of {max_samples} samples for classical models...")
            indices = np.random.choice(len(self.X_train_processed), max_samples, replace=False)
            X_train_subset = self.X_train_processed[indices]
            y_train_subset = self.y_train[indices]
        else:
            X_train_subset = self.X_train_processed
            y_train_subset = self.y_train

        # Define base classifiers with class weights
        base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=20, class_weight='balanced',
                n_jobs=-1, random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000, class_weight='balanced',
                n_jobs=-1, random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, max_depth=6,
                n_jobs=-1, random_state=self.random_state
            )
        }

        # Train individual models
        for name, base_model in base_models.items():
            print(f"Training {name}...")

            try:
                # Wrap in MultiOutputClassifier for multi-label
                model = MultiOutputClassifier(base_model, n_jobs=-1)
                model.fit(X_train_subset, y_train_subset)

                # Store model
                self.models[name] = model
                print(f"Successfully trained {name}")
            except ValueError as e:
                print(f"Skipping {name} due to class imbalance issue: {str(e)[:100]}...")
                continue

            # Predict probabilities and optimize threshold
            try:
                y_pred_proba = np.array([
                    clf.predict_proba(self.X_val_processed)[:, 1] if hasattr(clf, 'predict_proba')
                    else clf.decision_function(self.X_val_processed)
                    for clf in model.estimators_
                ]).T
            except:
                y_pred_proba = model.predict(self.X_val_processed).astype(float)

            self.optimize_threshold(name, y_pred_proba, self.y_val)

    def train_correlation_models(self):
        """Train models that capture label correlations"""
        print("Training label correlation models...")

        # Use subset for correlation models
        max_samples = 10000
        if len(self.X_train_processed) > max_samples:
            indices = np.random.choice(len(self.X_train_processed), max_samples, replace=False)
            X_train_subset = self.X_train_processed[indices]
            y_train_subset = self.y_train[indices]
        else:
            X_train_subset = self.X_train_processed
            y_train_subset = self.y_train

        # Single Classifier Chain
        print("Training Classifier Chain...")
        cc = ClassifierChainModel(
            base_classifier=LogisticRegression(max_iter=1000, random_state=self.random_state)
        )
        cc.fit(X_train_subset, y_train_subset)
        self.models['classifier_chain'] = cc

        # Optimize threshold
        y_pred_proba = cc.predict_proba(self.X_val_processed)
        self.optimize_threshold('classifier_chain', y_pred_proba, self.y_val)

        # Ensemble Classifier Chains (commented out for faster training)
        # Uncomment if you need maximum accuracy at cost of training time
        # print("Training Ensemble Classifier Chains...")
        # ecc = EnsembleClassifierChains(
        #     base_classifier=LogisticRegression(max_iter=1000, random_state=self.random_state),
        #     n_chains=3
        # )
        # ecc.fit(X_train_subset, y_train_subset)
        # self.models['ensemble_chains'] = ecc
        #
        # # Optimize threshold
        # y_pred_proba = ecc.predict_proba(self.X_val_processed)
        # self.optimize_threshold('ensemble_chains', y_pred_proba, self.y_val)

    def create_meta_ensemble(self):
        """Create a meta-ensemble from all trained models"""
        print("Creating meta-ensemble...")

        # Get predictions from all models
        all_predictions = {}

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(self.X_val_processed)
            else:
                pred_proba = model.predict(self.X_val_processed).astype(float)

            all_predictions[name] = pred_proba

        # Simple averaging ensemble (adjusted weights for disabled ensemble_chains)
        ensemble_weights = {
            'enhanced_nn': 0.35,
            'random_forest': 0.25,
            'xgboost': 0.25,
            'logistic_regression': 0.15,
            'classifier_chain': 0.0,
            # 'ensemble_chains': 0.0  # Disabled for faster training
        }

        # Weighted average
        ensemble_pred = np.zeros_like(list(all_predictions.values())[0])
        total_weight = 0

        for name, pred in all_predictions.items():
            if name in ensemble_weights:
                weight = ensemble_weights[name]
                ensemble_pred += weight * pred
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        # Optimize threshold for ensemble
        self.optimize_threshold('meta_ensemble', ensemble_pred, self.y_val)

        return ensemble_pred

    def evaluate_all_models(self):
        """Comprehensive evaluation of all models"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)

        results = {}

        # Evaluate individual models
        for name, model in self.models.items():
            print(f"\nEvaluating {name.upper()}...")

            # Get predictions
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(self.X_val_processed)
            else:
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

        # Evaluate meta-ensemble
        print(f"\nEvaluating META-ENSEMBLE...")
        ensemble_pred = self.create_meta_ensemble()
        threshold = self.best_thresholds.get('meta_ensemble', 0.5)
        pred_binary = (ensemble_pred > threshold).astype(int)

        f2_score = fbeta_score(self.y_val, pred_binary, average='micro', beta=2, zero_division=0)
        f1_score = fbeta_score(self.y_val, pred_binary, average='micro', beta=1, zero_division=0)

        results['meta_ensemble'] = {
            'F2_Score': f2_score,
            'F1_Score': f1_score,
            'Threshold': threshold
        }

        print(f"F2 Score: {f2_score:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Optimal Threshold: {threshold:.3f}")

        # Summary table
        print("\n" + "="*80)
        print("FINAL SUMMARY OF ALL MODELS")
        print("="*80)
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('F2_Score', ascending=False)
        print(results_df.round(4))

        return results_df

    def generate_final_predictions(self, use_best_model=True):
        """Generate final predictions using the best performing model"""
        print("Generating final predictions...")

        # Load test data if available
        try:
            test_embeddings = np.load("data/test_data.npy")
            print(f"Loaded test data: {test_embeddings.shape}")

            # Apply same preprocessing to test data
            test_processed = test_embeddings
            if hasattr(self, 'scaler'):
                test_processed = self.scaler.transform(test_processed)
            if hasattr(self, 'pca_reducer') and self.pca_reducer:
                test_processed = self.pca_reducer.transform(test_processed)
            if hasattr(self, 'feature_selector') and self.feature_selector:
                test_processed = self.feature_selector.transform(test_processed)

        except FileNotFoundError:
            print("Test data not found. Using validation set for demonstration.")
            test_processed = self.X_val_processed

        # Use best model or ensemble
        if use_best_model and hasattr(self, 'best_model_name'):
            model_name = self.best_model_name
            model = self.models[model_name]
            threshold = self.best_thresholds[model_name]

            if hasattr(model, 'predict_proba'):
                test_pred_proba = model.predict_proba(test_processed)
            else:
                test_pred_proba = model.predict(test_processed).astype(float)
        else:
            # Use meta-ensemble
            print("Using meta-ensemble for final predictions...")
            test_pred_proba = self.create_meta_ensemble_predictions(test_processed)
            threshold = self.best_thresholds.get('meta_ensemble', 0.5)

        # Convert to binary predictions
        test_pred_binary = (test_pred_proba > threshold).astype(int)

        # Convert back to label format
        test_pred_labels = self.mlb.inverse_transform(test_pred_binary)
        formatted_predictions = [';'.join(labels) for labels in test_pred_labels]

        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': range(1, len(formatted_predictions) + 1),
            'labels': formatted_predictions
        })

        return submission_df, test_pred_proba, test_pred_binary

    def create_meta_ensemble_predictions(self, X_test):
        """Create meta-ensemble predictions for test data"""
        all_predictions = {}

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_test)
            else:
                pred_proba = model.predict(X_test).astype(float)

            all_predictions[name] = pred_proba

        # Weighted average
        ensemble_weights = {
            'enhanced_nn': 0.35,
            'random_forest': 0.25,
            'xgboost': 0.25,
            'logistic_regression': 0.15,
            'classifier_chain': 0.0,
            # 'ensemble_chains': 0.0  # Disabled for faster training
        }

        ensemble_pred = np.zeros_like(list(all_predictions.values())[0])
        total_weight = 0

        for name, pred in all_predictions.items():
            if name in ensemble_weights:
                weight = ensemble_weights[name]
                ensemble_pred += weight * pred
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        return ensemble_pred

    def save_all_models(self, save_dir="enhanced_models"):
        """Save all trained models and preprocessors"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving all models to {save_dir}/...")

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

        # Save thresholds and other metadata
        joblib.dump(self.best_thresholds, f"{save_dir}/best_thresholds.pkl")
        joblib.dump(self.imbalance_handler, f"{save_dir}/imbalance_handler.pkl")

        print("All models saved successfully!")

def main():
    """Main execution function for comprehensive enhanced classification"""
    print("COMPREHENSIVE ENHANCED MEDICAL CHART CLASSIFICATION")
    print("=" * 80)
    print("Integrating: Threshold Optimization + Class Imbalance Handling +")
    print("Label Correlation Modeling + Ensemble Methods + Feature Engineering")
    print("=" * 80)

    # Initialize classifier
    classifier = ComprehensiveEnhancedClassifier(random_state=42)

    # 1. Load and preprocess data
    print("\n1. DATA LOADING AND PREPROCESSING")
    print("-" * 40)
    X, y = classifier.load_and_preprocess_data()

    # 2. Split data
    print("\n2. DATA SPLITTING")
    print("-" * 40)
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data()

    # 3. Feature engineering
    print("\n3. FEATURE ENGINEERING")
    print("-" * 40)
    X_train_processed, X_val_processed, X_test_processed = classifier.apply_feature_engineering(
        apply_pca=True, pca_components=256,
        apply_scaling=True, apply_feature_selection=True, k_best=200
    )

    # 4. Handle class imbalance
    print("\n4. CLASS IMBALANCE HANDLING")
    print("-" * 40)
    X_train_processed, y_train = classifier.handle_class_imbalance(
        apply_sampling=True, apply_class_weights=True
    )

    # 5. Train all models
    print("\n5. MODEL TRAINING")
    print("-" * 40)

    # Enhanced Neural Network
    nn_model, history = classifier.train_enhanced_neural_network(epochs=12, batch_size=64)

    # Classical ML Ensemble
    classifier.train_classical_ensemble()

    # Label Correlation Models
    classifier.train_correlation_models()

    # 6. Comprehensive evaluation
    print("\n6. COMPREHENSIVE EVALUATION")
    print("-" * 40)
    results = classifier.evaluate_all_models()

    # 7. Generate final predictions
    print("\n7. FINAL PREDICTIONS")
    print("-" * 40)
    submission_df, pred_proba, pred_binary = classifier.generate_final_predictions()

    # Save submission file
    submission_df.to_csv("enhanced_submission.csv", index=False)
    print("Enhanced submission saved to 'enhanced_submission.csv'")

    # 8. Save all models
    print("\n8. SAVING MODELS")
    print("-" * 40)
    classifier.save_all_models()

    print("\n" + "="*80)
    print("COMPREHENSIVE ENHANCED CLASSIFICATION COMPLETED!")
    print("="*80)
    print(f"Best model: {results.index[0]} with F2 score: {results.iloc[0]['F2_Score']:.4f}")

    return classifier, results

if __name__ == "__main__":
    classifier, results = main()