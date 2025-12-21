import os
import glob
import pandas as pd
from ultralytics import YOLO
import torch
import numpy as np
import random
import yaml
import shutil

# Configuration
WEIGHTS_DIR = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\Weights"
DATA_YAML = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\data\dataset.yaml"
OUTPUT_DIR = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\Inference_results"
SEEDS = [42, 123, 2024, 7, 999]  # Use specific seeds for reproducibility
NUM_RUNS = len(SEEDS)
SUBSET_FRACTION = 0.9  # Use 90% of the test set for each run to get variation

def run_inference():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all .pt files in Weights directory
    weight_files = glob.glob(os.path.join(WEIGHTS_DIR, "*.pt"))
    print(f"Found {len(weight_files)} weight files: {[os.path.basename(f) for f in weight_files]}")
    
    if not weight_files:
        print(f"No weight files found in {WEIGHTS_DIR}")
        return

    # Load original dataset.yaml
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get all test images
    test_dir = os.path.join(data_config['path'], data_config['test'])
    test_images = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
    
    if not test_images:
        print(f"No test images found in {test_dir}")
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
            
            for i, seed in enumerate(SEEDS):
                print(f"  Run {i+1}/{NUM_RUNS} with seed {seed}...")
                
                # Set seeds
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Create a random subset of test images
                subset_size = int(len(test_images) * SUBSET_FRACTION)
                subset = random.sample(test_images, subset_size)
                
                # Write subset to a temporary file
                temp_txt = os.path.join(model_run_dir, f"test_run_{i+1}.txt")
                with open(temp_txt, 'w') as f:
                    for img_path in subset:
                        f.write(img_path + '\n')
                
                # Create a temporary dataset.yaml
                temp_yaml_path = os.path.join(model_run_dir, f"dataset_run_{i+1}.yaml")
                temp_config = data_config.copy()
                temp_config['test'] = temp_txt
                with open(temp_yaml_path, 'w') as f:
                    yaml.dump(temp_config, f)
                
                # Run validation
                metrics = model.val(
                    data=temp_yaml_path, 
                    split='test', 
                    verbose=False,
                    project=model_run_dir,
                    name=f"run_{i+1}",
                    exist_ok=True,
                    half=True
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
                
                # Cleanup temporary files
                os.remove(temp_txt)
                os.remove(temp_yaml_path)

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
