# run_phase2.py - Save this in swiggy-ml/ folder (same level as requirements.txt)
import sys
import os
sys.path.append('src')

from data_validation import load_and_validate_data
from eda_helpers import quick_analysis
from utils import logger
from config import *

def main():
    print("🚀 Starting Phase 2: Data Acquisition & Initial Analysis")
    print("="*60)
    
    # Load and validate data
    df = load_and_validate_data()
    
    if not df.empty:
        # Run quick analysis
        quick_analysis(df)
        
        print("\n✅ Phase 2 completed successfully!")
        print(f"📁 Check the following directories:")
        print(f"   - Figures: {FIGURES_DIR}")
        print(f"   - Reports: {REPORTS_DIR}")
        print(f"   - Processed Data: {DATA_PROCESSED}")
        
        # List generated files
        print(f"\n📊 Generated Files:")
        figures = list(FIGURES_DIR.glob('*.png'))
        reports = list(REPORTS_DIR.glob('*'))
        
        for file in figures:
            print(f"   - {file.name}")
        for file in reports:
            print(f"   - {file.name}")
            
    else:
        print("❌ Phase 2 failed - no data available")

if __name__ == "__main__":
    main()