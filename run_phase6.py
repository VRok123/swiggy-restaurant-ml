# run_phase6.py
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import logger, setup_plotting, save_model
from config import *

class AdvancedModeler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
    
    def create_leakage_free_datasets(self):
        """Create separate feature sets for each target to avoid data leakage"""
        logger.info("Creating leakage-free datasets...")
        
        restaurants_df = pd.read_csv(DATA_PROCESSED / "swiggy_restaurants_enhanced.csv")
        df = restaurants_df.copy()
        
        # Create basic engineered features
        df['price_to_dish_ratio'] = df['avg_price'] / (df['dish_count'] + 1)
        df['rating_count_per_dish'] = df['total_rating_count'] / (df['dish_count'] + 1)
        df['has_high_variance'] = (df['rating_std'] > df['rating_std'].median()).astype(int)
        df['price_volatility'] = df['price_std'] / (df['avg_price'] + 1)
        
        # One-hot encoding for city (top cities only)
        top_cities = df['city'].value_counts().head(10).index
        city_dummies = pd.get_dummies(df['city'], prefix='city')
        city_columns_to_keep = [f'city_{city}' for city in top_cities if f'city_{city}' in city_dummies.columns]
        if 'city_other' not in city_dummies.columns:
            # Create 'other' category for remaining cities
            other_cities = [col for col in city_dummies.columns if col not in city_columns_to_keep]
            if other_cities:
                city_dummies['city_other'] = city_dummies[other_cities].max(axis=1)
                city_columns_to_keep.append('city_other')
        else:
            city_columns_to_keep.append('city_other')
        
        city_dummies = city_dummies[city_columns_to_keep]
        city_dummies = city_dummies.loc[:, ~city_dummies.columns.duplicated()]
        
        df = pd.concat([df, city_dummies], axis=1)
        city_columns = city_dummies.columns.tolist()
        
        # FEATURE SETS FOR EACH TASK (NO DATA LEAKAGE)
        
        # 1. HIGH RATED PREDICTION: Don't use rating-related features that define the target
        high_rated_features = [
            # Safe features (don't directly define high-rated)
            'avg_price', 'dish_count', 'total_rating_count', 
            'price_std', 'category_diversity', 'price_to_dish_ratio',
            'rating_count_per_dish', 'has_high_variance', 'price_volatility'
        ] + city_columns
        
        # 2. POPULAR PREDICTION: Don't use rating_count related features that define the target
        popular_features = [
            # Safe features (don't directly define popular)
            'avg_price', 'dish_count', 'avg_rating', 'median_rating',
            'price_std', 'category_diversity', 'price_to_dish_ratio',
            'has_high_variance', 'price_volatility'
        ] + city_columns
        
        # 3. PREMIUM PREDICTION: Don't use price-related features that define the target
        premium_features = [
            # Safe features (don't directly define premium)
            'dish_count', 'avg_rating', 'median_rating', 'total_rating_count',
            'rating_std', 'category_diversity', 'rating_count_per_dish',
            'has_high_variance'
        ] + city_columns
        
        # Create feature matrices
        X_high = df[high_rated_features].copy()
        X_pop = df[popular_features].copy()
        X_prem = df[premium_features].copy()
        
        # Handle missing values
        X_high = X_high.fillna(X_high.median())
        X_pop = X_pop.fillna(X_pop.median())
        X_prem = X_prem.fillna(X_prem.median())
        
        # Targets
        y_high = df['is_high_rated']
        y_pop = df['is_popular']
        y_prem = df['is_premium']
        
        logger.info(f"Leakage-free datasets created:")
        logger.info(f"  High-rated: {X_high.shape}")
        logger.info(f"  Popular: {X_pop.shape}")
        logger.info(f"  Premium: {X_prem.shape}")
        
        print(f"\n🔒 Leakage-Free Feature Sets:")
        print(f"   High-rated: {len(high_rated_features)} features (excludes rating features)")
        print(f"   Popular: {len(popular_features)} features (excludes rating_count features)")
        print(f"   Premium: {len(premium_features)} features (excludes price features)")
        
        return {
            'high_rated': (X_high, y_high, high_rated_features),
            'popular': (X_pop, y_pop, popular_features),
            'premium': (X_prem, y_prem, premium_features)
        }
    
    def create_advanced_models(self):
        """Define advanced models with optimized hyperparameters"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            )
        }
        return models
    
    def train_models(self, datasets):
        """Train advanced models for all tasks"""
        logger.info("Training advanced models...")
        
        models = self.create_advanced_models()
        results = {}
        
        for task_name, (X, y, feature_names) in datasets.items():
            logger.info(f"Training models for: {task_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            
            task_results = {}
            
            for model_name, model in models.items():
                logger.info(f"  - {model_name}")
                
                try:
                    # Scale features for tree-based models (optional but can help)
                    if model_name in ['xgboost', 'lightgbm']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    # Train model
                    model.fit(X_tr, y_train)
                    
                    # Predictions
                    y_pred_proba = model.predict_proba(X_te)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    
                    # Calculate comprehensive metrics
                    metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, task_name)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='roc_auc')
                    metrics['cv_roc_auc_mean'] = cv_scores.mean()
                    metrics['cv_roc_auc_std'] = cv_scores.std()
                    metrics['cv_roc_auc_scores'] = cv_scores
                    
                    # Store feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[f"{task_name}_{model_name}"] = {
                            'importances': model.feature_importances_,
                            'features': feature_names
                        }
                    
                    # Store results
                    task_results[model_name] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': {
                            'y_true': y_test,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba
                        },
                        'feature_names': feature_names
                    }
                    
                    # Store the model
                    self.models[f"{task_name}_{model_name}"] = model
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {task_name}: {e}")
                    continue
            
            results[task_name] = task_results
        
        return results
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, task_name):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC scores
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC (especially important for imbalanced data)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (recall_score(y_true, y_pred, pos_label=1) + 
                                      recall_score(y_true, y_pred, pos_label=0)) / 2
        
        # Additional metrics for business context
        metrics['positive_rate'] = y_true.mean()
        metrics['n_positive'] = y_true.sum()
        metrics['n_negative'] = len(y_true) - y_true.sum()
        
        return metrics
    
    def analyze_feature_importance(self):
        """Analyze and plot feature importance for all models"""
        logger.info("Analyzing feature importance...")
        
        if not self.feature_importance:
            logger.warning("No feature importance data available")
            return
        
        n_plots = len(self.feature_importance)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (model_key, importance_data) in enumerate(self.feature_importance.items()):
            if idx < len(axes):
                importances = importance_data['importances']
                features = importance_data['features']
                
                # Ensure arrays are the same length
                min_length = min(len(importances), len(features))
                importances = importances[:min_length]
                features = features[:min_length]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=True).tail(15)  # Top 15 features
                
                # Plot
                axes[idx].barh(importance_df['feature'], importance_df['importance'])
                axes[idx].set_title(f'Feature Importance: {model_key}', fontweight='bold', fontsize=12)
                axes[idx].set_xlabel('Importance')
                axes[idx].grid(axis='x', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(self.feature_importance), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'advanced_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance(self, results):
        """Plot comprehensive model performance comparison"""
        tasks = list(results.keys())
        metrics_to_plot = ['roc_auc', 'f1', 'pr_auc', 'balanced_accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                plot_data = []
                for task in tasks:
                    for model_name, model_results in results[task].items():
                        if metric in model_results['metrics']:
                            plot_data.append({
                                'Task': task.replace('_', ' ').title(),
                                'Model': model_name.replace('_', ' ').title(),
                                'Metric': metric.replace('_', ' ').title(),
                                'Score': model_results['metrics'][metric]
                            })
                
                if plot_data:
                    plot_df = pd.DataFrame(plot_data)
                    sns.barplot(data=plot_df, x='Task', y='Score', hue='Model', ax=axes[idx])
                    axes[idx].set_title(f'{metric.replace("_", " ").title()} by Task', 
                                      fontweight='bold', fontsize=12)
                    axes[idx].set_ylabel(metric.replace('_', ' ').title())
                    axes[idx].set_xlabel('')
                    axes[idx].tick_params(axis='x', rotation=45)
                    axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'advanced_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, results):
        """Plot ROC curves for all models and tasks"""
        from sklearn.metrics import roc_curve
        
        tasks = list(results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, task in enumerate(tasks):
            if idx < len(axes):
                ax = axes[idx]
                
                for model_name, model_results in results[task].items():
                    y_true = model_results['predictions']['y_true']
                    y_pred_proba = model_results['predictions']['y_pred_proba']
                    
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    roc_auc = model_results['metrics']['roc_auc']
                    
                    ax.plot(fpr, tpr, linewidth=2, 
                           label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title(f'ROC Curve - {task.replace("_", " ").title()}', 
                           fontweight='bold', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'advanced_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, results):
        """Plot Precision-Recall curves for all models and tasks"""
        tasks = list(results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, task in enumerate(tasks):
            if idx < len(axes):
                ax = axes[idx]
                
                for model_name, model_results in results[task].items():
                    y_true = model_results['predictions']['y_true']
                    y_pred_proba = model_results['predictions']['y_pred_proba']
                    
                    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                    pr_auc = model_results['metrics']['pr_auc']
                    
                    ax.plot(recall, precision, linewidth=2,
                           label=f'{model_name} (AUC = {pr_auc:.3f})')
                
                positive_rate = model_results['metrics']['positive_rate']
                ax.axhline(y=positive_rate, color='red', linestyle='--', alpha=0.7,
                          label=f'Baseline (Positive Rate = {positive_rate:.3f})')
                
                ax.set_xlabel('Recall', fontsize=12)
                ax.set_ylabel('Precision', fontsize=12)
                ax.set_title(f'Precision-Recall - {task.replace("_", " ").title()}', 
                           fontweight='bold', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'advanced_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, results):
        """Generate comprehensive model evaluation report"""
        logger.info("Generating comprehensive model report...")
        
        report_data = []
        
        for task in results:
            for model_name, model_results in results[task].items():
                metrics = model_results['metrics']
                
                report_data.append({
                    'Task': task,
                    'Model': model_name,
                    'ROC_AUC': f"{metrics['roc_auc']:.3f}",
                    'PR_AUC': f"{metrics['pr_auc']:.3f}",
                    'F1_Score': f"{metrics['f1']:.3f}",
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'Balanced_Accuracy': f"{metrics['balanced_accuracy']:.3f}",
                    'CV_ROC_AUC_Mean': f"{metrics['cv_roc_auc_mean']:.3f}",
                    'CV_ROC_AUC_Std': f"{metrics['cv_roc_auc_std']:.3f}",
                    'Positive_Rate': f"{metrics['positive_rate']:.3f}",
                    'N_Positive': metrics['n_positive'],
                    'N_Negative': metrics['n_negative']
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Save detailed report
        report_file = REPORTS_DIR / "advanced_model_comprehensive_report.csv"
        report_df.to_csv(report_file, index=False)
        logger.info(f"Comprehensive report saved to: {report_file}")
        
        # Print summary report
        print("\n📊 ADVANCED MODEL PERFORMANCE SUMMARY (No Data Leakage)")
        print("=" * 100)
        print(f"{'Task':<12} {'Model':<15} {'ROC-AUC':<8} {'F1-Score':<8} {'Accuracy':<8} {'CV ROC-AUC':<12}")
        print("-" * 100)
        
        for task in results:
            for model_name, model_results in results[task].items():
                metrics = model_results['metrics']
                print(f"{task:<12} {model_name:<15} {metrics['roc_auc']:.3f}    {metrics['f1']:.3f}     "
                      f"{metrics['accuracy']:.3f}     {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
        
        return report_df

def main():
    print("🚀 Starting Phase 6: Advanced Models with Proper Data Leakage Handling")
    print("=" * 70)
    
    try:
        setup_plotting()
        
        # Initialize advanced modeler
        modeler = AdvancedModeler()
        
        # Create leakage-free datasets
        print("\n🔒 Creating Leakage-Free Datasets...")
        datasets = modeler.create_leakage_free_datasets()
        
        # Train advanced models
        print("\n🤖 Training Advanced Models...")
        results = modeler.train_models(datasets)
        
        # Generate comprehensive report
        print("\n📈 Generating Comprehensive Report...")
        report_df = modeler.generate_comprehensive_report(results)
        
        # Create visualizations
        print("\n🎨 Creating Visualizations...")
        modeler.plot_model_performance(results)
        modeler.plot_roc_curves(results)
        modeler.plot_precision_recall_curves(results)
        modeler.analyze_feature_importance()
        
        # Save all models
        print("\n💾 Saving Trained Models...")
        for model_name, model in modeler.models.items():
            save_model(model, f"advanced_{model_name}.pkl")
        
        print("\n✅ PHASE 6 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Final summary
        print(f"\n🎯 BEST PERFORMING MODELS:")
        for task in results:
            if results[task]:
                best_roc = max(results[task].items(), key=lambda x: x[1]['metrics']['roc_auc'])
                best_f1 = max(results[task].items(), key=lambda x: x[1]['metrics']['f1'])
                print(f"   {task.upper().replace('_', ' ')}:")
                print(f"      Best ROC-AUC: {best_roc[0]} ({best_roc[1]['metrics']['roc_auc']:.3f})")
                print(f"      Best F1-Score: {best_f1[0]} ({best_f1[1]['metrics']['f1']:.3f})")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"   - Model performance: figures/advanced_model_performance.png")
        print(f"   - ROC curves: figures/advanced_roc_curves.png")
        print(f"   - Precision-Recall curves: figures/advanced_precision_recall_curves.png")
        print(f"   - Feature importance: figures/advanced_feature_importance.png")
        print(f"   - Comprehensive report: reports/advanced_model_comprehensive_report.csv")
        print(f"   - Trained models: models/advanced_*.pkl")
        
        print(f"\n🔒 DATA LEAKAGE PREVENTION:")
        print(f"   ✓ High-rated prediction: Excludes rating features")
        print(f"   ✓ Popular prediction: Excludes rating_count features")
        print(f"   ✓ Premium prediction: Excludes price features")
        
    except Exception as e:
        logger.error(f"❌ Phase 6 failed: {e}")
        raise

if __name__ == "__main__":
    main()