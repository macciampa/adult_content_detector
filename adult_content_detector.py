import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

class AdultContentDetector:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.max_length = 100
        
        # Initialize models
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        }
        
        self.trained_models = {}
        self.neural_network = None
        
    def load_data(self, file_path):
        """Load data from Excel file"""
        print("Loading data...")
        self.df = pd.read_excel(file_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Category distribution:\n{self.df['Category'].value_counts()}")
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("\nPreprocessing data...")
        
        # Convert categories to binary (0: Non_Adult, 1: Adult)
        self.df['Category_binary'] = (self.df['Category'] == 'Adult').astype(int)
        
        # Clean text data
        self.df['Description_clean'] = self.df['Description'].str.lower()
        self.df['Description_clean'] = self.df['Description_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
        self.df['Description_clean'] = self.df['Description_clean'].str.replace(r'\s+', ' ', regex=True)
        self.df['Description_clean'] = self.df['Description_clean'].str.strip()
        
        print("Data preprocessing completed.")
        return self.df
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        print(f"\nSplitting data (test_size={test_size}, val_size={val_size})...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.df['Description_clean'],
            self.df['Category_binary'],
            test_size=test_size,
            random_state=42,
            stratify=self.df['Category_binary']
        )
        
        # Second split: separate validation set from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size/(1-test_size),
            random_state=42,
            stratify=y_temp
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_tfidf_features(self):
        """Create TF-IDF features for traditional ML models"""
        print("\nCreating TF-IDF features...")
        
        # Fit TF-IDF on training data
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        self.X_val_tfidf = self.tfidf_vectorizer.transform(self.X_val)
        self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
        
        print(f"TF-IDF feature matrix shape: {self.X_train_tfidf.shape}")
        return self.X_train_tfidf, self.X_val_tfidf, self.X_test_tfidf
    
    def create_neural_network_features(self):
        """Create features for neural network"""
        print("\nCreating neural network features...")
        
        # Fit tokenizer on training data
        self.tokenizer.fit_on_texts(self.X_train)
        
        # Convert text to sequences
        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_val_seq = self.tokenizer.texts_to_sequences(self.X_val)
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        
        # Pad sequences
        self.X_train_padded = pad_sequences(self.X_train_seq, maxlen=self.max_length, padding='post', truncating='post')
        self.X_val_padded = pad_sequences(self.X_val_seq, maxlen=self.max_length, padding='post', truncating='post')
        self.X_test_padded = pad_sequences(self.X_test_seq, maxlen=self.max_length, padding='post', truncating='post')
        
        print(f"Neural network feature matrix shape: {self.X_train_padded.shape}")
        return self.X_train_padded, self.X_val_padded, self.X_test_padded
    
    def train_traditional_models(self):
        """Train traditional ML models"""
        print("\nTraining traditional ML models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train_tfidf, self.y_train)
            self.trained_models[name] = model
            
            # Validation performance
            val_pred = model.predict(self.X_val_tfidf)
            val_accuracy = accuracy_score(self.y_val, val_pred)
            print(f"{name} validation accuracy: {val_accuracy:.4f}")
    
    def build_neural_network(self):
        """Build and train neural network"""
        print("\nBuilding and training neural network...")
        
        vocab_size = len(self.tokenizer.word_index) + 1
        
        self.neural_network = keras.Sequential([
            layers.Embedding(vocab_size, 128, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.neural_network.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.neural_network.fit(
            self.X_train_padded, self.y_train,
            epochs=20,
            batch_size=32,
            validation_data=(self.X_val_padded, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model and return metrics"""
        # Predictions
        if model_name == 'Neural Network':
            y_pred_proba = model.predict(X_test).flatten()
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auc_pr': auc_pr,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
    
    def get_feature_importance(self, model, model_name):
        """Get feature importance for models that support it"""
        if model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'feature_importances'):
                return model.feature_importances
        elif model_name == 'Logistic Regression':
            if hasattr(model, 'coef_'):
                return np.abs(model.coef_[0])
        return None
    
    def plot_performance_metrics(self, results):
        """Plot performance metrics (first image)"""
        print("\nGenerating performance metrics plot...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auc_pr']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (model_name, result) in enumerate(results.items()):
            values = [result[metric] for metric in metrics]
            axes[0, 0].bar(x + i*width, values, width, label=model_name, alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(metric_names, rotation=45)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        for model_name, result in results.items():
            axes[0, 1].plot(result['fpr'], result['tpr'], 
                           label=f"{model_name} (AUC = {result['auc']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        for model_name, result in results.items():
            axes[1, 0].plot(result['recall_curve'], result['precision_curve'], 
                           label=f"{model_name} (AUC-PR = {result['auc_pr']:.3f})")
        
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (for models that support it)
        importance_data = []
        importance_labels = []
        
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
            if model_name in self.trained_models:
                importance = self.get_feature_importance(self.trained_models[model_name], model_name)
                if importance is not None:
                    # Get top 10 features
                    top_indices = np.argsort(importance)[-10:]
                    top_importance = importance[top_indices]
                    top_features = [self.tfidf_vectorizer.get_feature_names_out()[i] for i in top_indices]
                    
                    importance_data.extend(top_importance)
                    importance_labels.extend([f"{model_name}: {feat}" for feat in top_features])
        
        if importance_data:
            axes[1, 1].barh(range(len(importance_data)), importance_data)
            axes[1, 1].set_yticks(range(len(importance_labels)))
            axes[1, 1].set_yticklabels(importance_labels, fontsize=8)
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top Features by Model')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models (second image)"""
        print("\nGenerating confusion matrices plot...")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Calculate grid dimensions for 5 models
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        
        # Flatten axes if there's only one row
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot confusion matrix for each model
        for i, (model_name, result) in enumerate(results.items()):
            row = i // n_cols
            col = i % n_cols
            
            cm = result['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Adult', 'Adult'],
                       yticklabels=['Non-Adult', 'Adult'],
                       ax=axes[row, col])
            
            axes[row, col].set_title(f'{model_name}\nConfusion Matrix')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
            
            # Add metrics text
            accuracy = result['accuracy']
            precision = result['precision']
            recall = result['recall']
            f1 = result['f1']
            
            textstr = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            axes[row, col].text(0.02, 0.98, textstr, transform=axes[row, col].transAxes, 
                               fontsize=9, verticalalignment='top', bbox=props)
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('model_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results(self, results):
        """Plot comprehensive results (legacy method - now calls both separate methods)"""
        print("\nGenerating plots...")
        self.plot_performance_metrics(results)
        self.plot_confusion_matrices(results)
    
    def print_results(self, results):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create results table
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auc_pr']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
        
        print(f"{'Model':<20}", end="")
        for metric in metric_names:
            print(f"{metric:<12}", end="")
        print()
        print("-" * 92)
        
        for model_name, result in results.items():
            print(f"{model_name:<20}", end="")
            for metric in metrics:
                print(f"{result[metric]:<12.4f}", end="")
            print()
        
        print("\n" + "="*80)
        print("DETAILED RESULTS BY MODEL")
        print("="*80)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}")
            print("-" * len(model_name))
            print(f"Accuracy:  {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall:    {result['recall']:.4f}")
            print(f"F1-Score:  {result['f1']:.4f}")
            print(f"AUC-ROC:   {result['auc']:.4f}")
            print(f"AUC-PR:    {result['auc_pr']:.4f}")
            
            print("\nConfusion Matrix:")
            cm = result['confusion_matrix']
            print(f"                Predicted")
            print(f"                Non-Adult  Adult")
            print(f"Actual Non-Adult    {cm[0,0]:<8}  {cm[0,1]:<5}")
            print(f"      Adult         {cm[1,0]:<8}  {cm[1,1]:<5}")
    
    def run_complete_pipeline(self, file_path):
        """Run the complete pipeline"""
        print("ADULT CONTENT DETECTION PIPELINE")
        print("="*50)
        
        # Load and preprocess data
        self.load_data(file_path)
        self.preprocess_data()
        
        # Split data
        self.split_data()
        
        # Create features
        self.create_tfidf_features()
        self.create_neural_network_features()
        
        # Train traditional models
        self.train_traditional_models()
        
        # Train neural network
        history = self.build_neural_network()
        
        # Evaluate all models
        print("\nEvaluating models on test set...")
        results = {}
        
        # Evaluate traditional models
        for name, model in self.trained_models.items():
            results[name] = self.evaluate_model(model, self.X_test_tfidf, self.y_test, name)
        
        # Evaluate neural network
        results['Neural Network'] = self.evaluate_model(
            self.neural_network, self.X_test_padded, self.y_test, 'Neural Network'
        )
        
        # Print and plot results
        self.print_results(results)
        self.plot_results(results)
        
        return results

if __name__ == "__main__":
    # Initialize and run the pipeline
    detector = AdultContentDetector()
    results = detector.run_complete_pipeline('adult_content.xlsx') 