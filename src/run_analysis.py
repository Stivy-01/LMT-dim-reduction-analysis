#!/usr/bin/env python3
"""
Main script to run the LMT behavioral analysis pipeline.
"""

from pathlib import Path
from analysis.gui import AnalysisGUI
from analysis.analysis_pipeline import AnalysisPipeline

def main():
    # 1. Show GUI to get configuration
    print("Starting LMT Behavioral Analysis...")
    gui = AnalysisGUI()
    config = gui.run()
    
    if config is None:
        print("Analysis cancelled by user.")
        return
    
    # 2. Create and run pipeline
    print(f"\nAnalyzing dataset: {Path(config['dataset_path']).name}")
    pipeline = AnalysisPipeline(config['output_dir'])
    results = pipeline.run_analysis(
        config['dataset_path'],
        analysis_type=config['analysis_type']
    )
    
    # 3. Show completion message
    print("\n✅ Analysis complete!")
    print(f"Results saved in: {config['output_dir']}")
    print("\nGenerated files:")
    print("- Analysis report (HTML)")
    print("- Analysis report (Markdown)")
    print("- Results CSV files")
    print("- Statistical summaries")

if __name__ == "__main__":
    main() 