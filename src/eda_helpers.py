import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import List, Dict, Any
from utils import logger
from config import *

class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def analyze_numeric_distributions(self, columns: List[str] = None):
        """Analyze distributions of numeric columns"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.df.columns]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for distribution analysis")
            return
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                self.df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue')
                axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'numeric_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return numeric_cols
    
    def analyze_categorical_distributions(self, columns: List[str] = None, top_n: int = 15):
        """Analyze distributions of categorical columns"""
        if columns is None:
            cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        else:
            cat_cols = [col for col in columns if col in self.df.columns]
        
        if not cat_cols:
            logger.warning("No categorical columns found for distribution analysis")
            return
        
        for col in cat_cols:
            value_counts = self.df[col].value_counts().head(top_n)
            
            plt.figure(figsize=(12, 6))
            bars = value_counts.plot(kind='bar', color='lightcoral', alpha=0.7)
            plt.title(f'Top {top_n} Values in {col}', fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for i, (idx, val) in enumerate(value_counts.items()):
                plt.text(i, val + 0.01 * max(value_counts), str(val), 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'categorical_{col}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return cat_cols

def quick_analysis(df: pd.DataFrame):
    """Quick comprehensive analysis of the dataset"""
    analyzer = EDAAnalyzer(df)
    
    print("🔍 Quick Analysis Results:")
    print("="*40)
    
    # Numeric analysis
    numeric_cols = analyzer.analyze_numeric_distributions()
    if numeric_cols:
        print(f"📊 Numeric columns analyzed: {numeric_cols}")
    
    # Categorical analysis
    cat_cols = analyzer.analyze_categorical_distributions()
    if cat_cols:
        print(f"📝 Categorical columns analyzed: {cat_cols}")
    
    # Correlation analysis (if multiple numeric columns)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📈 Correlation matrix generated")