import os
import pandas as pd
from ultralytics import YOLO
import yaml
import glob

# Configuration
MODELS_TO_TEST = [
    "yolov5s.pt",
    "yolov8s.pt",
    "yolov9s.pt",
    "yolov10s.pt",
    "yolo11s.pt",
    "yolo12s.pt" 
]
SEEDS = [42] 
DATA_YAML = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\data\dataset.yaml"
PROJECT_DIR = r"c:\Users\Bioresources\Desktop\GitHub\AgriScout-Beetle-Detection\runs\train_comparison"

# Training Arguments (Hyperparameters + Augmentation)
TRAIN_ARGS = {
    # Training params
    "epochs": 200,
    "batch": -1, # Adjust based on your GPU memory
    "imgsz": 1280,
    "patience": 50,
    "lr0": 0.01,
    "box": 5.0,
    
    # Augmentation params
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
}

def get_best_metrics(run_dir):
    """Extract best mAP metrics from results.csv"""
    try:
        # Find results.csv
        csv_files = glob.glob(os.path.join(run_dir, 'results.csv'))
        if not csv_files:
            return None
        
        df = pd.read_csv(csv_files[0])
        # Column names usually have spaces, strip them
        df.columns = [c.strip() for c in df.columns]
        
        # Get the row with best fitness or just the last epoch? 
        # Usually we want the best mAP50-95. 
        # Ultralytics saves best.pt based on fitness.
        # Let's take the max mAP50-95
        
        if 'metrics/mAP50-95(B)' in df.columns:
            best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
            return best_row.to_dict()
        return None
    except Exception as e:
        print(f"Error reading metrics from {run_dir}: {e}")
        return None

def train_models():
    results_summary = []
    
    # Ensure project directory exists
    os.makedirs(PROJECT_DIR, exist_ok=True)

    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*20}\nStarting training for model: {model_name}\n{'='*20}")
        model_results = []
        
        for seed in SEEDS:
            run_name = f"{model_name.replace('.pt', '')}_seed{seed}"
            print(f"  Running seed: {seed}")
            
            try:
                # Load model
                # If the model file doesn't exist locally, it will try to download it.
                # If the model name is invalid, it will raise an error.
                model = YOLO(model_name)
                
                # Train
                results = model.train(
                    data=DATA_YAML,
                    project=PROJECT_DIR,
                    name=run_name,
                    seed=seed,
                    exist_ok=True, # Overwrite if exists (though seeds should make it unique)
                    **TRAIN_ARGS
                )
                
                # Get metrics
                save_dir = results.save_dir
                metrics = get_best_metrics(save_dir)
                
                if metrics:
                    metrics['model'] = model_name
                    metrics['seed'] = seed
                    metrics['dir'] = save_dir
                    model_results.append(metrics)
                    print(f"    Finished {run_name}: mAP50={metrics.get('metrics/mAP50(B)', 0):.4f}, mAP50-95={metrics.get('metrics/mAP50-95(B)', 0):.4f}")
                else:
                    print(f"    Finished {run_name} but could not retrieve metrics.")
                    model_results.append({
                        'model': model_name,
                        'seed': seed,
                        'error': 'No metrics found'
                    })
                
            except Exception as e:
                print(f"Failed to train {model_name} with seed {seed}: {e}")
                model_results.append({
                    'model': model_name,
                    'seed': seed,
                    'error': str(e)
                })

        # Compute average for this model
        valid_results = [r for r in model_results if 'error' not in r]
        if valid_results:
            # Create a DataFrame from the valid results
            df_results = pd.DataFrame(valid_results)
            
            # Calculate mean for numeric columns only
            numeric_cols = df_results.select_dtypes(include=['number']).columns
            avg_metrics = df_results[numeric_cols].mean().to_dict()
            
            # Add model name and run count
            avg_metrics['model'] = model_name
            avg_metrics['runs'] = len(valid_results)
            
            print(f"\nAverage results for {model_name}:")
            for k, v in avg_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            
            results_summary.append(avg_metrics)
            
    # Save summary
    if results_summary:
        df = pd.DataFrame(results_summary)
        # Reorder columns to put model and runs first
        cols = ['model', 'runs'] + [c for c in df.columns if c not in ['model', 'runs']]
        df = df[cols]
        
        summary_path = os.path.join(PROJECT_DIR, 'model_comparison_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(df)

if __name__ == "__main__":
    train_models()
