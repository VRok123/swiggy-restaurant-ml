# run_phase5.py
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc)
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import logger, setup_plotting, save_model
from config import *

class BaselineModeler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load the prepared modeling data"""
        logger.info("Loading modeling data...")
        
        # Load features and targets
        features_df = pd.read_csv(DATA_PROCESSED / "model_features.csv")
        restaurants_df = pd.read_csv(DATA_PROCESSED / "swiggy_restaurants_enhanced.csv")
        
        X = features_df.values
        y_high = restaurants_df['is_high_rated']
        y_popular = restaurants_df['is_popular']
        y_premium = restaurants_df['is_premium']
        
        feature_names = features_df.columns.tolist()
        
        logger.info(f"Loaded data: X {X.shape}, features: {len(feature_names)}")
        return X, y_high, y_popular, y_premium, feature_names
    
    def create_splits(self, X, y_high, y_popular, y_premium):
        """Create train-test splits for all targets"""
        splits = {}
        
        # High-rated classification (imbalanced)
        splits['high_rated'] = train_test_split(
            X, y_high, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=y_high
        )
        
        # Popular classification
        splits['popular'] = train_test_split(
            X, y_popular, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y_popular
        )
        
        # Premium classification
        splits['premium'] = train_test_split(
            X, y_premium, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y_premium
        )
        
        return splits
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train_baseline_models(self, splits, feature_names):
        """Train baseline models for all classification tasks"""
        logger.info("Training baseline models...")
        
        tasks = ['high_rated', 'popular', 'premium']
        models_to_train = {
            'dummy_stratified': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE),
            'dummy_most_frequent': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
            'logistic': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
        }
        
        results = {}
        
        for task in tasks:
            logger.info(f"Training models for: {task}")
            X_train, X_test, y_train, y_test = splits[task]
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            task_results = {}
            
            for model_name, model in models_to_train.items():
                logger.info(f"  - {model_name}")
                
                # Use scaled features for linear models, original for tree-based
                if model_name in ['logistic']:
                    X_tr, X_te = X_train_scaled, X_test_scaled
                else:
                    X_tr, X_te = X_train, X_test
                
                # Train model
                model.fit(X_tr, y_train)
                
                # Predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_te)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_te)
                    y_pred_proba = None
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, task)
                
                # Store results
                task_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': {
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                # Store the actual model for important ones
                if model_name in ['logistic', 'random_forest']:
                    self.models[f"{task}_{model_name}"] = model
            
            results[task] = task_results
        
        return results
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba, task_name):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC if probabilities available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-Recall AUC (especially important for imbalanced data)
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Class distribution info
        metrics['positive_rate'] = y_true.mean()
        metrics['n_positive'] = y_true.sum()
        metrics['n_negative'] = len(y_true) - y_true.sum()
        
        return metrics
    
    def analyze_feature_importance(self, results, feature_names):
        """Analyze feature importance from trained models"""
        logger.info("Analyzing feature importance...")
        
        importance_results = {}
        
        for task in ['high_rated', 'popular', 'premium']:
            if f"{task}_logistic" in self.models:
                # Logistic regression coefficients
                model = self.models[f"{task}_logistic"]
                coefficients = model.coef_[0]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
                
                importance_results[f"{task}_logistic"] = importance_df
            
            if f"{task}_random_forest" in self.models:
                # Random Forest feature importance
                model = self.models[f"{task}_random_forest"]
                importances = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_results[f"{task}_random_forest"] = importance_df
        
        return importance_results
    
    def plot_results(self, results, importance_results):
        """Plot model results and feature importance"""
        self.plot_model_comparison(results)
        self.plot_feature_importance(importance_results)
    
    def plot_model_comparison(self, results):
        """Plot comparison of model performance across tasks"""
        tasks = ['high_rated', 'popular', 'premium']
        metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                # Prepare data for plotting
                plot_data = []
                for task in tasks:
                    for model_name, model_results in results[task].items():
                        plot_data.append({
                            'Task': task.replace('_', ' ').title(),
                            'Model': model_name.replace('_', ' ').title(),
                            'Metric': metric.upper(),
                            'Score': model_results['metrics'][metric]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                # Create plot
                sns.barplot(data=plot_df, x='Task', y='Score', hue='Model', ax=axes[idx])
                axes[idx].set_title(f'{metric.upper()} by Task and Model', fontweight='bold')
                axes[idx].set_ylabel(metric.upper())
                axes[idx].set_xlabel('')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'baseline_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_results):
        """Plot feature importance for different models and tasks"""
        n_plots = len(importance_results)
        if n_plots == 0:
            return
        
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (model_name, importance_df) in enumerate(importance_results.items()):
            if idx < len(axes):
                # Plot top 10 features
                top_features = importance_df.head(10)
                
                if 'coefficient' in importance_df.columns:
                    # Logistic regression - plot coefficients
                    colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
                    axes[idx].barh(top_features['feature'], top_features['coefficient'], color=colors)
                    axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    axes[idx].set_xlabel('Coefficient Value')
                else:
                    # Random Forest - plot importance
                    axes[idx].barh(top_features['feature'], top_features['importance'])
                    axes[idx].set_xlabel('Feature Importance')
                
                axes[idx].set_title(f'Feature Importance: {model_name}', fontweight='bold')
                axes[idx].set_ylabel('Features')
                axes[idx].grid(axis='x', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(importance_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_importance_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results, importance_results):
        """Generate comprehensive model report"""
        logger.info("Generating model report...")
        
        report_data = []
        
        for task in ['high_rated', 'popular', 'premium']:
            for model_name, model_results in results[task].items():
                metrics = model_results['metrics']
                
                report_data.append({
                    'Task': task,
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1']:.3f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.3f}",
                    'PR-AUC': f"{metrics['pr_auc']:.3f}",
                    'Positive_Rate': f"{metrics['positive_rate']:.3f}"
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_file = REPORTS_DIR / "baseline_model_report.csv"
        report_df.to_csv(report_file, index=False)
        logger.info(f"Model report saved to: {report_file}")
        
        # Print summary
        print("\n📊 Baseline Model Performance Summary:")
        print("=" * 80)
        print(report_df.to_string(index=False))
        
        return report_df

def main():
    print("🚀 Starting Phase 5: Baseline Models & Evaluation")
    print("=" * 60)
    
    try:
        setup_plotting()
        
        # Initialize modeler
        modeler = BaselineModeler()
        
        # Load data
        X, y_high, y_popular, y_premium, feature_names = modeler.load_data()
        
        # Create splits
        splits = modeler.create_splits(X, y_high, y_popular, y_premium)
        
        # Train baseline models
        results = modeler.train_baseline_models(splits, feature_names)
        
        # Analyze feature importance
        importance_results = modeler.analyze_feature_importance(results, feature_names)
        
        # Generate plots
        modeler.plot_results(results, importance_results)
        
        # Generate report
        report_df = modeler.generate_report(results, importance_results)
        
        # Save models
        for model_name, model in modeler.models.items():
            save_model(model, f"baseline_{model_name}.pkl")
        
        print("\n✅ Phase 5 completed successfully!")
        print(f"📁 Generated Files:")
        print(f"   - Model comparison: figures/baseline_model_comparison.png")
        print(f"   - Feature importance: figures/feature_importance_baseline.png")
        print(f"   - Model report: reports/baseline_model_report.csv")
        print(f"   - Trained models: models/baseline_*.pkl")
        
        # Key insights
        print(f"\n🎯 Key Insights:")
        for task in ['high_rated', 'popular', 'premium']:
            best_model = max(results[task].items(), 
                           key=lambda x: x[1]['metrics']['roc_auc'])
            print(f"   {task}: Best model = {best_model[0]} (ROC-AUC: {best_model[1]['metrics']['roc_auc']:.3f})")
        
    except Exception as e:
        logger.error(f"❌ Phase 5 failed: {e}")
        raise

if __name__ == "__main__":
    main()