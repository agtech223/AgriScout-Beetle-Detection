import os
import glob
import pandas as pd
from ultralytics import YOLO
import torch
import numpy as np

# Configuration
WEIGHTS_DIR = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\Weights"
DATA_YAML = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\data\dataset.yaml"
OUTPUT_DIR = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\Inference_results"
NUM_RUNS = 10
SEEDS = [42, 123, 2024, 7, 999]  # Use specific seeds for reproducibility

def run_inference():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all .pt files in Weights directory
    weight_files = glob.glob(os.path.join(WEIGHTS_DIR, "*.pt"))
    print(f"Found {len(weight_files)} weight files: {[os.path.basename(f) for f in weight_files]}")
    
    if not weight_files:
        print(f"No weight files found in {WEIGHTS_DIR}")
        return

    all_results = []

    for weight_path in weight_files:
        model_name = os.path.basename(weight_path)
        model_stem = os.path.splitext(model_name)[0]
        print(f"Processing model: {model_name}")
        
        # Create model specific directory
        model_run_dir = os.path.join(OUTPUT_DIR, model_stem)
        os.makedirs(model_run_dir, exist_ok=True)
        
        model_results = []
        
        try:
            # Load model
            model = YOLO(weight_path)
            
            for i in range(NUM_RUNS):
                seed = SEEDS[i] if i < len(SEEDS) else i
                print(f"  Run {i+1}/{NUM_RUNS} with seed {seed}...")
                
                # Set seed globally just in case
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                # Run validation
                # project=model_run_dir, name=f"run_{i+1}" will save results to model_run_dir/run_{i+1}
                metrics = model.val(
                    data=DATA_YAML, 
                    split='test', 
                    verbose=False,
                    project=model_run_dir,
                    name=f"run_{i+1}",
                    exist_ok=True
                )
                
                r = metrics.results_dict
                
                run_data = {
                    'Model': model_name,
                    'Run': i + 1,
                    'Seed': seed,
                    'Precision': r.get('metrics/precision(B)', 0),
                    'Recall': r.get('metrics/recall(B)', 0),
                    'mAP50': r.get('metrics/mAP50(B)', 0),
                    'mAP50-95': r.get('metrics/mAP50-95(B)', 0)
                }
                model_results.append(run_data)
                all_results.append(run_data)

            # Compute mean for this model
            df_model = pd.DataFrame(model_results)
            mean_metrics = df_model[['Precision', 'Recall', 'mAP50', 'mAP50-95']].mean()
            
            mean_data = {
                'Model': model_name,
                'Run': 'Mean',
                'Seed': 'N/A',
                'Precision': mean_metrics['Precision'],
                'Recall': mean_metrics['Recall'],
                'mAP50': mean_metrics['mAP50'],
                'mAP50-95': mean_metrics['mAP50-95']
            }
            
            # Append mean to model results for saving
            model_results.append(mean_data)
            all_results.append(mean_data)
            
            # Save model specific CSV
            df_model_final = pd.DataFrame(model_results)
            model_csv_path = os.path.join(model_run_dir, f"{model_stem}_results.csv")
            df_model_final.to_csv(model_csv_path, index=False)
            print(f"  Saved results for {model_name} to {model_csv_path}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # Save all results to CSV
    if all_results:
        df_final = pd.DataFrame(all_results)
        output_path = os.path.join(OUTPUT_DIR, 'inference_comparison_results.csv')
        df_final.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Also save a summary with just the means
        df_means = df_final[df_final['Run'] == 'Mean']
        summary_path = os.path.join(OUTPUT_DIR, 'inference_summary_means.csv')
        df_means.to_csv(summary_path, index=False)
        print(f"Summary means saved to {summary_path}")

if __name__ == "__main__":
    run_inference()
