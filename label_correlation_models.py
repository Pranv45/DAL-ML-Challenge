#!/usr/bin/env python3
"""
Label Correlation Models for Multi-label Medical Classification
Implements methods to capture dependencies between labels
"""

import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score, accuracy_score, hamming_loss
from sklearn.model_selection import cross_val_score
import itertools
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class LabelCorrelationAnalyzer:
    """Analyzes correlations between labels in multi-label data"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.cooccurrence_matrix = None
        self.label_pairs = None
        
    def analyze_correlations(self, y_multilabel, label_names=None):
        """Analyze correlations between labels"""
        print("Analyzing label correlations...")
        
        # Calculate Pearson correlation between labels
        self.correlation_matrix = np.corrcoef(y_multilabel.T)
        
        # Calculate co-occurrence matrix
        self.cooccurrence_matrix = np.dot(y_multilabel.T, y_multilabel)
        
        # Find highly correlated label pairs
        n_labels = y_multilabel.shape[1]
        correlations = []
        
        for i in range(n_labels):
            for j in range(i+1, n_labels):
                corr = self.correlation_matrix[i, j]
                cooccur = self.cooccurrence_matrix[i, j]
                total_i = np.sum(y_multilabel[:, i])
                total_j = np.sum(y_multilabel[:, j])
                
                # Calculate conditional probabilities
                if total_i > 0:
                    prob_j_given_i = cooccur / total_i
                else:
                    prob_j_given_i = 0
                    
                if total_j > 0:
                    prob_i_given_j = cooccur / total_j
                else:
                    prob_i_given_j = 0
                
                correlations.append({
                    'label_i': i,
                    'label_j': j,
                    'correlation': corr,
                    'cooccurrence': cooccur,
                    'prob_j_given_i': prob_j_given_i,
                    'prob_i_given_j': prob_i_given_j,
                    'label_i_name': label_names[i] if label_names else f'Label_{i}',
                    'label_j_name': label_names[j] if label_names else f'Label_{j}'
                })
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        self.label_pairs = correlations
        
        # Print top correlations
        print(f"\nTop 10 strongest correlations:")
        for i, corr in enumerate(correlations[:10]):
            print(f"{i+1:2d}. {corr['label_i_name']} - {corr['label_j_name']}: "
                  f"r={corr['correlation']:.3f}, co-occur={corr['cooccurrence']}")
        
        return correlations
    
    def get_optimal_chain_order(self, y_multilabel, method='frequency'):
        """Determine optimal order for classifier chains"""
        n_labels = y_multilabel.shape[1]
        
        if method == 'frequency':
            # Order by label frequency (most frequent first)
            label_counts = y_multilabel.sum(axis=0)
            order = np.argsort(label_counts)[::-1]
            
        elif method == 'correlation':
            # Order by average correlation with other labels
            if self.correlation_matrix is None:
                self.analyze_correlations(y_multilabel)
            
            avg_correlations = np.mean(np.abs(self.correlation_matrix), axis=1)
            order = np.argsort(avg_correlations)[::-1]
            
        elif method == 'conditional_dependency':
            # Order by conditional dependencies (most dependent on others first)
            dependencies = []
            for i in range(n_labels):
                # Calculate how much this label depends on others
                dependency_score = 0
                for j in range(n_labels):
                    if i != j:
                        # Mutual information approximation
                        dependency_score += abs(self.correlation_matrix[i, j])
                dependencies.append(dependency_score)
            
            order = np.argsort(dependencies)[::-1]
        
        else:
            # Random order
            np.random.seed(42)
            order = np.random.permutation(n_labels)
        
        return order

class ClassifierChainModel:
    """Implements Classifier Chains for multi-label classification"""
    
    def __init__(self, base_classifier=None, chain_order=None, random_state=42):
        self.base_classifier = base_classifier or LogisticRegression(random_state=random_state)
        self.chain_order = chain_order
        self.random_state = random_state
        self.models = []
        self.analyzer = LabelCorrelationAnalyzer()
        
    def fit(self, X, y_multilabel):
        """Train classifier chains"""
        print("Training Classifier Chains...")
        
        n_labels = y_multilabel.shape[1]
        
        # Determine chain order if not provided
        if self.chain_order is None:
            self.analyzer.analyze_correlations(y_multilabel)
            self.chain_order = self.analyzer.get_optimal_chain_order(y_multilabel, method='correlation')
        
        print(f"Chain order: {self.chain_order[:10]}..." if len(self.chain_order) > 10 else f"Chain order: {self.chain_order}")
        
        # Train classifiers in chain order
        self.models = []
        for i, label_idx in enumerate(self.chain_order):
            print(f"Training classifier {i+1}/{n_labels} for label {label_idx}")
            
            # Prepare features: original features + previous predictions
            if i == 0:
                X_extended = X
            else:
                # Add predictions from previous classifiers in chain
                prev_predictions = np.zeros((X.shape[0], i))
                for j, prev_label_idx in enumerate(self.chain_order[:i]):
                    prev_predictions[:, j] = y_multilabel[:, prev_label_idx]
                X_extended = np.hstack([X, prev_predictions])
            
            # Train classifier for current label
            y_current = y_multilabel[:, label_idx]
            model = self._create_base_classifier()
            model.fit(X_extended, y_current)
            self.models.append(model)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities for all labels"""
        n_samples = X.shape[0]
        n_labels = len(self.models)
        
        predictions = np.zeros((n_samples, n_labels))
        probabilities = np.zeros((n_samples, n_labels))
        
        # Make predictions in chain order
        for i, (model, label_idx) in enumerate(zip(self.models, self.chain_order)):
            if i == 0:
                X_extended = X
            else:
                # Add previous predictions
                prev_predictions = predictions[:, :i]
                X_extended = np.hstack([X, prev_predictions])
            
            # Get probabilities for current label
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_extended)
                if prob.shape[1] == 2:  # Binary classification
                    probabilities[:, label_idx] = prob[:, 1]
                else:
                    probabilities[:, label_idx] = prob[:, 0]
            else:
                # For models without predict_proba, use decision function or predict
                try:
                    scores = model.decision_function(X_extended)
                    # Convert to probabilities using sigmoid
                    probabilities[:, label_idx] = 1 / (1 + np.exp(-scores))
                except:
                    probabilities[:, label_idx] = model.predict(X_extended)
            
            # Update predictions for next classifier in chain
            predictions[:, label_idx] = (probabilities[:, label_idx] > 0.5).astype(int)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels"""
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)
    
    def _create_base_classifier(self):
        """Create a copy of the base classifier"""
        if hasattr(self.base_classifier, 'get_params'):
            params = self.base_classifier.get_params()
            classifier_class = type(self.base_classifier)
            return classifier_class(**params)
        else:
            return self.base_classifier

class EnsembleClassifierChains:
    """Ensemble of Classifier Chains with different orders"""
    
    def __init__(self, base_classifier=None, n_chains=5, random_state=42):
        self.base_classifier = base_classifier or LogisticRegression(random_state=random_state)
        self.n_chains = n_chains
        self.random_state = random_state
        self.chains = []
        
    def fit(self, X, y_multilabel):
        """Train multiple classifier chains with different orders"""
        print(f"Training ensemble of {self.n_chains} classifier chains...")
        
        n_labels = y_multilabel.shape[1]
        self.chains = []
        
        # Analyze correlations once
        analyzer = LabelCorrelationAnalyzer()
        analyzer.analyze_correlations(y_multilabel)
        
        for i in range(self.n_chains):
            print(f"\nTraining chain {i+1}/{self.n_chains}")
            
            # Use different ordering methods
            if i == 0:
                order = analyzer.get_optimal_chain_order(y_multilabel, method='frequency')
            elif i == 1:
                order = analyzer.get_optimal_chain_order(y_multilabel, method='correlation')
            elif i == 2:
                order = analyzer.get_optimal_chain_order(y_multilabel, method='conditional_dependency')
            else:
                # Random orders for remaining chains
                np.random.seed(self.random_state + i)
                order = np.random.permutation(n_labels)
            
            # Create and train chain
            chain = ClassifierChainModel(
                base_classifier=self.base_classifier,
                chain_order=order,
                random_state=self.random_state + i
            )
            chain.fit(X, y_multilabel)
            self.chains.append(chain)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities by averaging across chains"""
        if not self.chains:
            raise ValueError("Model must be fitted first")
        
        # Average predictions from all chains
        all_probabilities = []
        for chain in self.chains:
            prob = chain.predict_proba(X)
            all_probabilities.append(prob)
        
        # Average probabilities
        avg_probabilities = np.mean(all_probabilities, axis=0)
        return avg_probabilities
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels"""
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)

class LabelPowersetModel:
    """Implements Label Powerset for multi-label classification"""
    
    def __init__(self, base_classifier=None, min_examples=2, random_state=42):
        self.base_classifier = base_classifier or LogisticRegression(random_state=random_state)
        self.min_examples = min_examples
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.label_combinations = None
        
    def fit(self, X, y_multilabel):
        """Train label powerset model"""
        print("Training Label Powerset model...")
        
        # Convert multi-label to label combinations
        label_combinations = []
        for i in range(y_multilabel.shape[0]):
            # Create string representation of label combination
            combo = tuple(np.where(y_multilabel[i] == 1)[0])
            label_combinations.append(combo)
        
        # Count combinations and filter rare ones
        combo_counts = Counter(label_combinations)
        valid_combos = {combo for combo, count in combo_counts.items() 
                       if count >= self.min_examples}
        
        print(f"Total unique combinations: {len(combo_counts)}")
        print(f"Valid combinations (>= {self.min_examples} examples): {len(valid_combos)}")
        
        # Filter data to only include valid combinations
        valid_indices = [i for i, combo in enumerate(label_combinations) 
                        if combo in valid_combos]
        
        if len(valid_indices) == 0:
            raise ValueError(f"No label combinations have >= {self.min_examples} examples")
        
        X_filtered = X[valid_indices]
        y_filtered = [label_combinations[i] for i in valid_indices]
        
        # Encode label combinations as single labels
        y_encoded = self.label_encoder.fit_transform(y_filtered)
        
        # Store label combinations for decoding
        self.label_combinations = {
            encoded: combo for combo, encoded in 
            zip(y_filtered, y_encoded)
        }
        
        # Train classifier
        self.classifier = self._create_base_classifier()
        self.classifier.fit(X_filtered, y_encoded)
        
        print(f"Trained on {len(X_filtered)} samples with {len(np.unique(y_encoded))} classes")
        return self
    
    def predict_proba(self, X):
        """Predict probabilities for all labels"""
        if self.classifier is None:
            raise ValueError("Model must be fitted first")
        
        n_samples = X.shape[0]
        n_labels = max(max(combo) for combo in self.label_combinations.values()) + 1
        
        # Get class probabilities
        if hasattr(self.classifier, 'predict_proba'):
            class_probabilities = self.classifier.predict_proba(X)
        else:
            # For models without predict_proba, use one-hot encoding of predictions
            predictions = self.classifier.predict(X)
            n_classes = len(self.label_encoder.classes_)
            class_probabilities = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                if pred < n_classes:
                    class_probabilities[i, pred] = 1.0
        
        # Convert class probabilities to label probabilities
        label_probabilities = np.zeros((n_samples, n_labels))
        
        for sample_idx in range(n_samples):
            for class_idx, prob in enumerate(class_probabilities[sample_idx]):
                if class_idx < len(self.label_encoder.classes_):
                    encoded_combo = self.label_encoder.classes_[class_idx]
                    if encoded_combo in self.label_combinations:
                        combo = self.label_combinations[encoded_combo]
                        for label_idx in combo:
                            if label_idx < n_labels:
                                label_probabilities[sample_idx, label_idx] += prob
        
        return label_probabilities
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels"""
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)
    
    def _create_base_classifier(self):
        """Create a copy of the base classifier"""
        if hasattr(self.base_classifier, 'get_params'):
            params = self.base_classifier.get_params()
            classifier_class = type(self.base_classifier)
            return classifier_class(**params)
        else:
            return self.base_classifier

def compare_correlation_models(X_train, y_train, X_val, y_val, base_classifier=None):
    """Compare different label correlation modeling approaches"""
    print("Comparing Label Correlation Models")
    print("=" * 50)
    
    if base_classifier is None:
        base_classifier = LogisticRegression(random_state=42, max_iter=1000)
    
    models = {}
    results = {}
    
    # 1. One-vs-Rest (baseline - no correlation modeling)
    print("\n1. Training One-vs-Rest (baseline)...")
    ovr = OneVsRestClassifier(base_classifier)
    ovr.fit(X_train, y_train)
    models['OneVsRest'] = ovr
    
    # 2. Single Classifier Chain
    print("\n2. Training Single Classifier Chain...")
    cc = ClassifierChainModel(base_classifier=base_classifier)
    cc.fit(X_train, y_train)
    models['ClassifierChain'] = cc
    
    # 3. Ensemble Classifier Chains
    print("\n3. Training Ensemble Classifier Chains...")
    ecc = EnsembleClassifierChains(base_classifier=base_classifier, n_chains=3)
    ecc.fit(X_train, y_train)
    models['EnsembleChains'] = ecc
    
    # 4. Label Powerset (if feasible)
    print("\n4. Training Label Powerset...")
    try:
        lp = LabelPowersetModel(base_classifier=base_classifier, min_examples=2)
        lp.fit(X_train, y_train)
        models['LabelPowerset'] = lp
    except Exception as e:
        print(f"Label Powerset failed: {e}")
        models['LabelPowerset'] = None
    
    # Evaluate all models
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for name, model in models.items():
        if model is None:
            continue
            
        print(f"\nEvaluating {name}...")
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_val)
        
        # Calculate metrics
        f2_score = fbeta_score(y_val, y_pred, average='micro', beta=2, zero_division=0)
        f1_score = fbeta_score(y_val, y_pred, average='micro', beta=1, zero_division=0)
        hamming = hamming_loss(y_val, y_pred)
        
        results[name] = {
            'F2_Score': f2_score,
            'F1_Score': f1_score,
            'Hamming_Loss': hamming
        }
        
        print(f"F2 Score: {f2_score:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY OF ALL CORRELATION MODELS")
    print("="*60)
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('F2_Score', ascending=False)
    print(results_df.round(4))
    
    return models, results_df

def demonstrate_correlation_modeling():
    """Demonstrate label correlation modeling techniques"""
    print("Demonstrating Label Correlation Modeling")
    print("=" * 50)
    
    # Create synthetic correlated multi-label data
    np.random.seed(42)
    n_samples = 2000
    n_features = 50
    n_labels = 15
    
    X = np.random.randn(n_samples, n_features)
    
    # Create correlated labels
    y_multilabel = np.zeros((n_samples, n_labels))
    
    # Base probabilities for each label
    base_probs = np.random.uniform(0.1, 0.4, n_labels)
    
    # Create some label correlations
    for i in range(n_samples):
        for j in range(n_labels):
            prob = base_probs[j]
            
            # Add correlations
            if j > 0 and y_multilabel[i, j-1] == 1:  # Previous label influence
                prob *= 2.0  # Increase probability if previous label is active
            if j > 1 and y_multilabel[i, j-2] == 1:  # Two labels back influence
                prob *= 1.5
            
            # Clip probability
            prob = min(prob, 0.8)
            
            y_multilabel[i, j] = np.random.binomial(1, prob)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_multilabel[:split_idx], y_multilabel[split_idx:]
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    
    # Analyze correlations
    analyzer = LabelCorrelationAnalyzer()
    correlations = analyzer.analyze_correlations(y_train)
    
    # Compare models
    models, results = compare_correlation_models(X_train, y_train, X_val, y_val)
    
    return analyzer, models, results

if __name__ == "__main__":
    analyzer, models, results = demonstrate_correlation_modeling()