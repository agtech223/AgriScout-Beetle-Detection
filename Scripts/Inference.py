import os
from ultralytics import YOLO

def run_benchmarks():
    # List of models to benchmark
    models_to_benchmark = [
        #'yolov5s.pt',
        #'yolov8s.pt',
        #'yolov9s.pt',
        #'yolov10s.pt',
        #'yolo11s.pt',
        'yolo12s.pt'
    ]

    # Path to your dataset yaml
    dataset_yaml = 'data/dataset.yaml'
    img_size = 1280
    
    if not os.path.exists(dataset_yaml):
        print(f"Dataset config {dataset_yaml} does not exist.")
        return

    for model_name in models_to_benchmark:
        print(f"\n--- Benchmarking {model_name} ---")
        try:
            # Load the model
            model = YOLO(model_name)

            # Benchmark on your dataset
            # Note: benchmark() by default checks multiple export formats. 
            # If you only want speed/accuracy on PyTorch, you might just want val()
            # but the user specifically asked for benchmark.
            results = model.benchmark(data=dataset_yaml, imgsz=img_size)
            
            print(f"Finished benchmarking {model_name}")
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")

if __name__ == "__main__":
    run_benchmarks()
