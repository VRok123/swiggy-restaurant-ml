import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils import logger, print_data_info
from config import *

class DataValidator:
    def __init__(self):
        self.expected_columns = {
            'restaurant_name': 'object',
            'cuisines': 'object', 
            'location': 'object',
            'city': 'object',
            'rating': 'float64',
            'votes': 'int64',
            'cost_for_two': 'float64',
            'delivery_time': 'object',
            'veg_only': 'bool'
        }
        
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        validation_report = {}
        
        # 1. Basic structure validation
        validation_report['shape'] = df.shape
        validation_report['columns'] = list(df.columns)
        validation_report['dtypes'] = df.dtypes.to_dict()
        
        # 2. Missing values analysis
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        validation_report['missing_values'] = {
            'counts': missing_data[missing_data > 0].to_dict(),
            'percentages': missing_pct[missing_pct > 0].to_dict()
        }
        
        # 3. Duplicate analysis
        validation_report['duplicates'] = {
            'total_duplicates': df.duplicated().sum(),
            'percentage_duplicates': (df.duplicated().sum() / len(df)) * 100
        }
        
        # 4. Data type validation
        type_issues = {}
        for col, expected_type in self.expected_columns.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    type_issues[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        validation_report['type_issues'] = type_issues
        
        # 5. Value range validation
        value_checks = {}
        if 'rating' in df.columns:
            value_checks['rating_range'] = {
                'min': df['rating'].min(),
                'max': df['rating'].max(),
                'valid_range': (0, 5)
            }
        
        if 'cost_for_two' in df.columns:
            value_checks['cost_range'] = {
                'min': df['cost_for_two'].min(),
                'max': df['cost_for_two'].max(),
                'outliers_high': len(df[df['cost_for_two'] > 10000]),  # Arbitrary high threshold
                'outliers_low': len(df[df['cost_for_two'] < 0])
            }
            
        validation_report['value_checks'] = value_checks
        
        # 6. Data quality score
        quality_score = self.calculate_quality_score(validation_report)
        validation_report['quality_score'] = quality_score
        
        return validation_report
    
    def calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Penalize for missing values
        missing_penalty = sum(report['missing_values']['percentages'].values()) * 0.5
        score -= min(missing_penalty, 30)  # Max 30% penalty for missing values
        
        # Penalize for duplicates
        duplicate_penalty = report['duplicates']['percentage_duplicates'] * 2
        score -= min(duplicate_penalty, 20)  # Max 20% penalty for duplicates
        
        # Penalize for type issues
        type_penalty = len(report['type_issues']) * 5
        score -= min(type_penalty, 20)
        
        return max(score, 0)
    
    def generate_validation_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive validation report"""
        logger.info("🔍 Starting Data Validation...")
        
        report = self.validate_data_quality(df)
        
        print("\n" + "="*60)
        print("📊 DATA VALIDATION REPORT")
        print("="*60)
        
        print(f"📈 Data Shape: {report['shape']}")
        print(f"🏆 Quality Score: {report['quality_score']:.1f}/100")
        
        print(f"\n📋 Columns ({len(report['columns'])}):")
        for col in report['columns']:
            print(f"  - {col} ({report['dtypes'][col]})")
        
        if report['missing_values']['counts']:
            print(f"\n❌ Missing Values:")
            for col, count in report['missing_values']['counts'].items():
                pct = report['missing_values']['percentages'][col]
                print(f"  - {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\n✅ No missing values found!")
        
        if report['duplicates']['total_duplicates'] > 0:
            print(f"\n⚠️  Duplicates: {report['duplicates']['total_duplicates']} "
                  f"({report['duplicates']['percentage_duplicates']:.1f}%)")
        else:
            print(f"\n✅ No duplicates found!")
        
        if report['type_issues']:
            print(f"\n🔧 Data Type Issues:")
            for col, issues in report['type_issues'].items():
                print(f"  - {col}: expected {issues['expected']}, got {issues['actual']}")
        else:
            print(f"\n✅ No data type issues found!")
        
        print("\n" + "="*60)
        
        # Save report
        report_df = pd.DataFrame([report])
        report_file = REPORTS_DIR / "data_validation_report.csv"
        report_df.to_csv(report_file, index=False)
        logger.info(f"Validation report saved to: {report_file}")

def load_and_validate_data() -> pd.DataFrame:
    """Load data and run validation"""
    validator = DataValidator()
    
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        logger.info(f"✅ Data loaded successfully: {df.shape}")
        
        # Run validation
        validator.generate_validation_report(df)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    load_and_validate_data()