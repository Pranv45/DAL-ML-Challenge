#!/usr/bin/env python3
"""
Class Imbalance Handling for Multi-label Medical Classification
Specialized techniques for handling imbalanced label distributions
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class MultiLabelImbalanceHandler:
    """Handles class imbalance in multi-label classification scenarios"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_weights = None
        self.sampling_strategy = None
        
    def analyze_label_distribution(self, y_multilabel, label_names=None):
        """Analyze the distribution of labels to identify imbalance"""
        print("Analyzing label distribution...")
        
        # Calculate label frequencies
        label_counts = y_multilabel.sum(axis=0)
        total_samples = len(y_multilabel)
        
        # Calculate statistics
        label_frequencies = label_counts / total_samples
        
        stats = {
            'total_labels': len(label_counts),
            'total_samples': total_samples,
            'min_frequency': label_frequencies.min(),
            'max_frequency': label_frequencies.max(),
            'mean_frequency': label_frequencies.mean(),
            'std_frequency': label_frequencies.std(),
            'labels_per_sample_avg': y_multilabel.sum(axis=1).mean(),
            'labels_per_sample_std': y_multilabel.sum(axis=1).std()
        }
        
        print(f"Total labels: {stats['total_labels']}")
        print(f"Average labels per sample: {stats['labels_per_sample_avg']:.2f}")
        print(f"Label frequency - Min: {stats['min_frequency']:.4f}, Max: {stats['max_frequency']:.4f}")
        print(f"Label frequency - Mean: {stats['mean_frequency']:.4f}, Std: {stats['std_frequency']:.4f}")
        
        # Identify severely imbalanced labels (less than 1% frequency)
        rare_labels = np.where(label_frequencies < 0.01)[0]
        common_labels = np.where(label_frequencies > 0.1)[0]
        
        print(f"Rare labels (< 1%): {len(rare_labels)}")
        print(f"Common labels (> 10%): {len(common_labels)}")
        
        # Create imbalance report
        imbalance_report = pd.DataFrame({
            'label_index': range(len(label_counts)),
            'count': label_counts,
            'frequency': label_frequencies,
            'category': ['rare' if i in rare_labels else 'common' if i in common_labels else 'medium' 
                        for i in range(len(label_counts))]
        })
        
        if label_names is not None:
            imbalance_report['label_name'] = label_names
        
        return stats, imbalance_report
    
    def compute_class_weights(self, y_multilabel, method='balanced'):
        """Compute class weights for imbalanced labels"""
        print(f"Computing class weights using method: {method}")
        
        n_labels = y_multilabel.shape[1]
        n_samples = y_multilabel.shape[0]
        
        if method == 'balanced':
            # Standard balanced class weights
            weights = []
            for i in range(n_labels):
                y_label = y_multilabel[:, i]
                pos_weight = n_samples / (2 * np.sum(y_label)) if np.sum(y_label) > 0 else 1.0
                neg_weight = n_samples / (2 * (n_samples - np.sum(y_label))) if np.sum(y_label) < n_samples else 1.0
                weights.append({'0': neg_weight, '1': pos_weight})
            
        elif method == 'balanced_subsample':
            # Balanced weights with subsampling consideration
            weights = []
            for i in range(n_labels):
                y_label = y_multilabel[:, i]
                class_counts = np.bincount(y_label.astype(int))
                if len(class_counts) == 2:
                    total = class_counts.sum()
                    pos_weight = total / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
                    neg_weight = total / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0
                    weights.append({'0': neg_weight, '1': pos_weight})
                else:
                    weights.append({'0': 1.0, '1': 1.0})
        
        elif method == 'focal':
            # Weights for focal loss (emphasize hard examples)
            weights = []
            for i in range(n_labels):
                y_label = y_multilabel[:, i]
                pos_ratio = np.mean(y_label)
                alpha = 1 - pos_ratio  # More weight for minority class
                weights.append({'0': 1-alpha, '1': alpha})
        
        self.label_weights = weights
        return weights
    
    def create_weighted_loss(self, class_weights, loss_type='binary_crossentropy'):
        """Create a weighted loss function for TensorFlow/Keras"""
        
        def weighted_binary_crossentropy(y_true, y_pred):
            """Custom weighted binary crossentropy loss"""
            # Clip predictions to prevent log(0)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            
            # Calculate weighted loss for each label
            losses = []
            for i in range(len(class_weights)):
                weight_0 = class_weights[i]['0']
                weight_1 = class_weights[i]['1']
                
                # Extract the i-th label
                y_true_i = y_true[:, i]
                y_pred_i = y_pred[:, i]
                
                # Calculate weighted binary crossentropy for this label
                loss_pos = weight_1 * y_true_i * tf.math.log(y_pred_i)
                loss_neg = weight_0 * (1 - y_true_i) * tf.math.log(1 - y_pred_i)
                loss_i = -(loss_pos + loss_neg)
                
                losses.append(loss_i)
            
            # Average across all labels
            total_loss = tf.reduce_mean(tf.stack(losses, axis=1))
            return total_loss
        
        def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
            """Focal loss for addressing class imbalance"""
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            
            # Calculate focal loss
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            
            focal_weight = alpha_t * tf.pow(1 - pt, gamma)
            focal_loss = -focal_weight * tf.math.log(pt)
            
            return tf.reduce_mean(focal_loss)
        
        if loss_type == 'weighted_binary_crossentropy':
            return weighted_binary_crossentropy
        elif loss_type == 'focal':
            return focal_loss
        else:
            return 'binary_crossentropy'
    
    def apply_multilabel_sampling(self, X, y_multilabel, method='mlsmote', k_neighbors=5):
        """Apply sampling techniques adapted for multi-label scenarios"""
        print(f"Applying multi-label sampling: {method}")
        
        # Convert multi-label to single label for sampling algorithms
        # by creating a unique label for each combination
        y_combined = []
        for i in range(len(y_multilabel)):
            # Create a string representation of the label combination
            label_combo = ''.join(map(str, y_multilabel[i].astype(int)))
            y_combined.append(label_combo)
        
        # Count label combinations
        combo_counts = Counter(y_combined)
        print(f"Unique label combinations: {len(combo_counts)}")
        print(f"Most common combinations: {combo_counts.most_common(5)}")
        
        # Apply sampling based on method
        if method == 'random_oversample':
            # Simple random oversampling of minority label combinations
            target_count = max(combo_counts.values())
            X_resampled, y_resampled = self._random_oversample_multilabel(
                X, y_multilabel, y_combined, target_count
            )
        
        elif method == 'random_undersample':
            # Random undersampling of majority combinations
            target_count = min([count for count in combo_counts.values() if count > 1])
            X_resampled, y_resampled = self._random_undersample_multilabel(
                X, y_multilabel, y_combined, target_count
            )
        
        elif method == 'smote_adapted':
            # SMOTE adapted for multi-label by treating each label independently
            X_resampled, y_resampled = self._smote_multilabel(X, y_multilabel, k_neighbors)
        
        elif method == 'label_specific_oversample':
            # Oversample based on individual label frequencies
            X_resampled, y_resampled = self._label_specific_oversample(X, y_multilabel)
        
        else:
            print(f"Unknown sampling method: {method}. Returning original data.")
            return X, y_multilabel
        
        print(f"Original dataset size: {X.shape}")
        print(f"Resampled dataset size: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def _random_oversample_multilabel(self, X, y_multilabel, y_combined, target_count):
        """Random oversampling for multi-label data"""
        from collections import defaultdict
        
        # Group indices by label combination
        combo_indices = defaultdict(list)
        for i, combo in enumerate(y_combined):
            combo_indices[combo].append(i)
        
        # Oversample minority combinations
        new_indices = []
        for combo, indices in combo_indices.items():
            count = len(indices)
            if count < target_count:
                # Oversample this combination
                oversample_count = target_count - count
                new_samples = np.random.choice(indices, oversample_count, replace=True)
                new_indices.extend(indices + list(new_samples))
            else:
                new_indices.extend(indices)
        
        return X[new_indices], y_multilabel[new_indices]
    
    def _random_undersample_multilabel(self, X, y_multilabel, y_combined, target_count):
        """Random undersampling for multi-label data"""
        from collections import defaultdict
        
        # Group indices by label combination
        combo_indices = defaultdict(list)
        for i, combo in enumerate(y_combined):
            combo_indices[combo].append(i)
        
        # Undersample majority combinations
        new_indices = []
        for combo, indices in combo_indices.items():
            count = len(indices)
            if count > target_count:
                # Undersample this combination
                sampled_indices = np.random.choice(indices, target_count, replace=False)
                new_indices.extend(sampled_indices)
            else:
                new_indices.extend(indices)
        
        return X[new_indices], y_multilabel[new_indices]
    
    def _smote_multilabel(self, X, y_multilabel, k_neighbors):
        """SMOTE adapted for multi-label classification"""
        # Apply SMOTE to each label independently and combine results
        X_resampled_all = []
        y_resampled_all = []
        
        for label_idx in range(y_multilabel.shape[1]):
            y_single_label = y_multilabel[:, label_idx]
            
            # Only apply SMOTE if minority class exists and has enough samples
            if np.sum(y_single_label) >= k_neighbors and np.sum(y_single_label) < len(y_single_label) - k_neighbors:
                try:
                    smote = SMOTE(random_state=self.random_state, k_neighbors=min(k_neighbors, np.sum(y_single_label)-1))
                    X_resampled_label, y_resampled_label = smote.fit_resample(X, y_single_label)
                    
                    # Store results for this label
                    X_resampled_all.append(X_resampled_label)
                    y_resampled_all.append(y_resampled_label)
                except:
                    # If SMOTE fails, use original data
                    X_resampled_all.append(X)
                    y_resampled_all.append(y_single_label)
            else:
                # Use original data if SMOTE cannot be applied
                X_resampled_all.append(X)
                y_resampled_all.append(y_single_label)
        
        # Find the maximum resampled size
        max_size = max(len(X_res) for X_res in X_resampled_all)
        
        # Use the resampled data from the most balanced label
        best_idx = np.argmax([len(X_res) for X_res in X_resampled_all])
        X_final = X_resampled_all[best_idx]
        
        # Reconstruct multi-label y by applying the same resampling pattern
        y_final = np.zeros((len(X_final), y_multilabel.shape[1]))
        for i in range(y_multilabel.shape[1]):
            if len(y_resampled_all[i]) == len(X_final):
                y_final[:, i] = y_resampled_all[i]
            else:
                # If sizes don't match, repeat the pattern
                original_size = len(y_multilabel)
                repeat_pattern = len(X_final) // original_size
                remainder = len(X_final) % original_size
                
                y_repeated = np.tile(y_multilabel[:, i], repeat_pattern)
                if remainder > 0:
                    y_repeated = np.concatenate([y_repeated, y_multilabel[:remainder, i]])
                y_final[:, i] = y_repeated
        
        return X_final, y_final.astype(int)
    
    def _label_specific_oversample(self, X, y_multilabel):
        """Oversample based on individual label frequencies"""
        # Calculate label frequencies
        label_counts = y_multilabel.sum(axis=0)
        target_count = int(np.median(label_counts))  # Use median as target
        
        # Find samples that need to be oversampled for each rare label
        samples_to_add = []
        labels_to_add = []
        
        for label_idx in range(y_multilabel.shape[1]):
            current_count = label_counts[label_idx]
            if current_count < target_count:
                # Find samples with this label
                positive_samples = np.where(y_multilabel[:, label_idx] == 1)[0]
                
                if len(positive_samples) > 0:
                    # Oversample these samples
                    oversample_count = int(target_count - current_count)
                    sampled_indices = np.random.choice(positive_samples, oversample_count, replace=True)
                    
                    samples_to_add.extend(sampled_indices)
                    labels_to_add.extend([label_idx] * oversample_count)
        
        # Combine original data with oversampled data
        if samples_to_add:
            X_oversampled = X[samples_to_add]
            y_oversampled = y_multilabel[samples_to_add]
            
            X_combined = np.vstack([X, X_oversampled])
            y_combined = np.vstack([y_multilabel, y_oversampled])
        else:
            X_combined = X
            y_combined = y_multilabel
        
        return X_combined, y_combined
    
    def create_balanced_class_weights_dict(self, y_multilabel):
        """Create class weight dictionary for sklearn models"""
        weights_dict = {}
        
        for i in range(y_multilabel.shape[1]):
            y_label = y_multilabel[:, i]
            
            # Calculate class weights for this label
            if np.sum(y_label) > 0 and np.sum(y_label) < len(y_label):
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=np.unique(y_label), 
                    y=y_label
                )
                weights_dict[i] = {0: class_weights[0], 1: class_weights[1]}
            else:
                weights_dict[i] = {0: 1.0, 1: 1.0}
        
        return weights_dict

def demonstrate_imbalance_handling():
    """Demonstrate the imbalance handling techniques"""
    print("Demonstrating Multi-label Imbalance Handling")
    print("=" * 50)
    
    # Create synthetic imbalanced multi-label data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    n_labels = 20
    
    X = np.random.randn(n_samples, n_features)
    
    # Create imbalanced labels (some labels much rarer than others)
    y_multilabel = np.zeros((n_samples, n_labels))
    
    # Common labels (appear in 20-50% of samples)
    for i in range(5):
        prob = np.random.uniform(0.2, 0.5)
        y_multilabel[:, i] = np.random.binomial(1, prob, n_samples)
    
    # Medium labels (appear in 5-20% of samples)
    for i in range(5, 15):
        prob = np.random.uniform(0.05, 0.2)
        y_multilabel[:, i] = np.random.binomial(1, prob, n_samples)
    
    # Rare labels (appear in 1-5% of samples)
    for i in range(15, 20):
        prob = np.random.uniform(0.01, 0.05)
        y_multilabel[:, i] = np.random.binomial(1, prob, n_samples)
    
    # Initialize handler
    handler = MultiLabelImbalanceHandler()
    
    # Analyze distribution
    stats, report = handler.analyze_label_distribution(y_multilabel)
    
    # Compute class weights
    weights = handler.compute_class_weights(y_multilabel, method='balanced')
    
    # Apply sampling
    X_resampled, y_resampled = handler.apply_multilabel_sampling(
        X, y_multilabel, method='label_specific_oversample'
    )
    
    # Compare distributions
    print("\nOriginal vs Resampled Label Distributions:")
    print("-" * 40)
    original_counts = y_multilabel.sum(axis=0)
    resampled_counts = y_resampled.sum(axis=0)
    
    for i in range(min(10, n_labels)):  # Show first 10 labels
        print(f"Label {i}: {int(original_counts[i]):4d} -> {int(resampled_counts[i]):4d}")
    
    return handler, X_resampled, y_resampled

if __name__ == "__main__":
    handler, X_res, y_res = demonstrate_imbalance_handling()