# AgriScout Beetle Detection: AI Agent Guide

## Project Overview
AgriScout is a **YOLOv12-based object detection system** for identifying beetles in agricultural imagery. The project implements a complete ML pipeline: training different YOLO variants, hyperparameter tuning, evaluation with multiple seeds, and inference benchmarking.

**Core Tech Stack**: PyTorch, Ultralytics YOLO (v8/v9/v10/v11/v12), Python 3.8+, pandas, OpenCV

## Architecture & Data Flow

### Four-Phase ML Pipeline
1. **Train.py** - Model training with seed variation and metric extraction
2. **Tuning.py** - Grid search over hyperparameters (lr0, box loss, augmentation params)
3. **Eval.py** - Validation/evaluation with 10 runs per model, reproducible seeding
4. **Inference.py** - Benchmarking models on the dataset

**Key Data Flow**: `data/dataset.yaml` → YOLO training → `runs/{train_comparison,hyperparam_tuning}` → metric aggregation to CSV

### Dataset Structure
- Defined by `data/dataset.yaml` (absolute paths in scripts - note Windows hardcoded paths that need adjustment)
- Images referenced as `1280x1280` (standard `imgsz`)
- YOLO format: separate train/val/test splits with corresponding annotations

## Critical Developer Workflows

### 1. Model Training
```bash
cd Scripts && python Train.py
# Trains models from MODELS_TO_TEST list with specific seeds
# Saves to: runs/train_comparison/{model_seed#}/weights/best.pt
```
- **Key params**: `TRAIN_ARGS` dict defines 200 epochs, batch=8, extensive augmentation
- **Reproducibility**: `SEEDS = [42]` ensures deterministic results
- **Key insight**: Extracts best mAP50-95 from `results.csv` via `get_best_metrics()` - handles column name stripping

### 2. Hyperparameter Tuning
```bash
python Tuning.py
# Grid searches 5 params across combinations (60+ configs)
# Saves to: grid_search_results.csv sorted by mAP50
```
- **Search space**: lr0, box, mosaic, scale, translate (see `param_grid` dict)
- **Quick eval**: 5 epochs per config for speed
- **Output**: Ranked CSV for identifying best hyperparameters

### 3. Evaluation with Reproducibility
```bash
python Eval.py
# 10 validation runs per model with seeds [42, 123, 2024, 7, 999]
# Computes mean/stddev across runs
```
- **Seed strategy**: Both `torch.manual_seed()` and `torch.cuda.manual_seed_all()` for GPU consistency
- **Metrics extracted**: Precision, Recall, mAP50, mAP50-95 from `model.val()` results_dict
- **Output structure**: Per-model CSV + aggregated `Inference_results` directory

### 4. Benchmarking
```bash
python Inference.py
# Runs model.benchmark() on dataset - tests speed across export formats
# Focus: inference performance, not training
```

## Code Patterns & Conventions

### Configuration as Constants
- Hardcoded paths at script top (Windows paths - **needs PATH abstraction**)
- Model names in lists: `["yolo12s.pt", "yolo11s.pt", ...]` (commented older versions)
- Seeds as lists for grid iteration

### Metric Extraction from YOLO
```python
metrics = model.val(data=DATA_YAML, ...)
results_dict = metrics.results_dict
mAP50_95 = results_dict['metrics/mAP50-95(B)']  # Key pattern
```
- Ultralytics stores metrics with `(B)` suffix for box detection
- Column names in `results.csv` have trailing spaces - strip them: `df.columns = [c.strip() for c in df.columns]`

### Directory Organization
- Results saved with descriptive names: `{model}_{seed}` or `grid_lr{lr0}_box{box}_...`
- CSV aggregation happens per-model then combined: `all_results.append(mean_data)`
- Output CSVs always include metadata columns (Model, Run, Seed) before metrics

### Reproducibility Priorities
- Seeds applied at multiple levels: script config + Ultralytics `seed=` param + torch manual seed
- Multiple runs (NUM_RUNS=10) to capture variance, mean computed for robustness
- `exist_ok=True` allows safe re-runs without deletion

## Integration Points & Dependencies

### External: Ultralytics YOLO Library
- **Load**: `model = YOLO(model_name)` auto-downloads pretrained weights if not cached
- **Train**: `model.train(data=DATA_YAML, project=PROJECT_DIR, name=run_name, **TRAIN_ARGS)`
- **Validate**: `model.val(data=DATA_YAML, split='test', verbose=False)` returns metrics object
- **Benchmark**: `model.benchmark(data=dataset_yaml, imgsz=img_size)` exports multiple formats

### File Inputs/Outputs
- **Input**: `data/dataset.yaml` (must have train/val/test splits defined)
- **Input**: Pre-trained weights from Ultralytics hub (auto-downloaded)
- **Output**: `runs/` directory auto-created by Ultralytics with dated subdirs
- **Output**: CSVs with results in project root and model-specific dirs

### Environment Dependencies
- GPU required (CUDA 13.0 specified in torch==2.9.1+cu130)
- 8GB+ VRAM recommended (batch=8 at 1280px resolution)
- Python path issues: Tuning.py uses relative `os.path` from script location

## Common Modifications & Extension Points

1. **Add models to train**: Edit `MODELS_TO_TEST` list in Train.py
2. **Adjust hyperparameter space**: Modify `param_grid` in Tuning.py
3. **Change resolution**: Update `imgsz: 1280` across scripts (search for "1280")
4. **Add metrics**: Extract additional keys from `results_dict` (e.g., `precision(B)`, `recall(B)`)
5. **Batch size**: Edit `TRAIN_ARGS["batch"]` based on GPU memory

## Known Issues & Paths to Fix

- **Windows hardcoded paths**: All scripts contain `r"c:\Users\Bioresources\..."` - needs `os.path` abstraction
- **Relative path fragility**: Tuning.py uses `os.path.dirname(os.path.dirname(...))` - break if script location changes
- **Missing check_setup.py**: README references but file not committed (environment validation)
- **No validation data splitting**: Scripts assume `dataset.yaml` properly splits train/val/test

## Quick Debugging Checklist

- `dataset.yaml` path exists and formatted correctly? (YOLO expects absolute or relative to script)
- GPU available? (`torch.cuda.is_available()` before running training)
- Weights directory exists for Eval.py? (looks in `Weights/` for `.pt` files)
- Results CSV column names have spaces? (Eval.py handles this - Train.py may need same fix)
