import os
from ultralytics import YOLO
from pathlib import Path

def run_benchmarks():
    # List of models to benchmark from the Weights folder
    models_to_benchmark = [
        'Weights/5s.pt',
        'Weights/8s.pt',
        'Weights/9s.pt',
        'Weights/10s.pt',
        'Weights/11s.pt',
        'Weights/12s.pt'
    ]

    # Path to your dataset yaml
    dataset_yaml = 'data/dataset.yaml'
    img_size = 1280
    
    if not os.path.exists(dataset_yaml):
        print(f"Dataset config {dataset_yaml} does not exist.")
        return

    for model_path in models_to_benchmark:
        print(f"\n--- Benchmarking {model_path} ---")
        try:
            # Load the model
            model = YOLO(model_path)

            # Run benchmark
            # Note: benchmark() runs performance tests across different formats.
            # It doesn't support 'project' and 'name' like predict() does.
            results = model.benchmark(data=dataset_yaml, imgsz=img_size)
            
            print(f"Finished benchmarking {model_path}")
        except Exception as e:
            print(f"Error benchmarking {model_path}: {e}")

if __name__ == "__main__":
    run_benchmarks()
