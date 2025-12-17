import itertools
import pandas as pd
from ultralytics import YOLO
import os

# Define paths
# Assuming script is in Scripts/ folder, so we go up one level to get to the root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATA_YAML = os.path.join(BASE_DIR, "data", "dataset.yaml")
PROJECT_DIR = os.path.join(BASE_DIR, "runs", "hyperparam_tuning")
CSV_PATH = os.path.join(BASE_DIR, "grid_search_results.csv")

if __name__ == '__main__':
    # Ensure project directory exists
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # Define your hyperparameter search space
    param_grid = {
        "lr0": [0.005, 0.01, 0.02],  # Learning rate
        "box": [5.0, 7.5, 10.0],     # Box loss gain
        "mosaic": [1.0, 0.8],        # Mosaic augmentation probability
        "scale": [0.5, 0.7],         # Scale factor for resizing images
        "translate": [0.1, 0.2],     # Small object jittering
    }

    results_list = []

    # Create all combinations of hyperparameter values
    for lr0, box, mosaic, scale, translate in itertools.product(
        param_grid["lr0"],
        param_grid["box"],
        param_grid["mosaic"],
        param_grid["scale"],
        param_grid["translate"]
    ):
        print(f"\nTraining with hyperparams: lr0={lr0}, box={box}, mosaic={mosaic}, scale={scale}, translate={translate}")

        # Initialize YOLO model
        model = YOLO("yolov8s.pt")

        # Train the model briefly (5 epochs) for quick testing
        model.train(
            data=DATA_YAML,                # Path to dataset.yaml
            epochs=5,                      # Short training for quick comparison
            imgsz=1280,                    # Higher resolution for small objects
            project=PROJECT_DIR,
            lr0=lr0,                       # Learning rate
            box=box,                       # Bounding-box loss gain
            mosaic=mosaic,                 # Mosaic augmentation probability
            scale=scale,                   # Scale transformation
            translate=translate,           # Small object jittering
            name=f"grid_lr{lr0}_box{box}_mos{mosaic}_scale{scale}_trans{translate}"
        )

        # After training, run validation to get metrics
        val_results = model.val(
            data=DATA_YAML,
            imgsz=1280,
            project=PROJECT_DIR,
            name=f"grid_lr{lr0}_box{box}_mos{mosaic}_scale{scale}_trans{translate}" # Re-use name to find the right run? model.val() usually runs on the loaded model.
        )

        # Extract mAP50 from validation results
        mAP50 = val_results.box.map50
        print(f"Validation mAP50: {mAP50}")

        # Store results
        results_list.append({
            "lr0": lr0,
            "box": box,
            "mosaic": mosaic,
            "scale": scale,
            "translate": translate,
            "mAP50": mAP50
        })

    # Convert results to a DataFrame and sort by mAP50
    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values("mAP50", ascending=False)
    print("\nGrid search results (sorted by mAP50):")
    print(df_sorted)

    # Save to CSV
    df_sorted.to_csv(CSV_PATH, index=False)
    print(f"\nResults saved to {CSV_PATH}")
