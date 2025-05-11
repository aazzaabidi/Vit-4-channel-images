

## Features

- **Preprocessing**: Handles raw time series → 2D encoded representations
- **Model Zoo**:
  - Vision Transformer (ViT)
  - Swin Transformer 
  - ResNet-Transformer hybrids
- **Training**:
  - Automatic mixed precision
  - Distributed training support
  - Hyperparameter tuning
- **Evaluation**:
  - Standard metrics (Accuracy, F1, Cohen's Kappa)
  - UMAP/T-SNE visualizations
  - Confusion matrices

## Installation

```bash
git clone https://github.com/yourusername/time-series-transformers.git
cd time-series-transformers
pip install -r requirements.txt
```

## Data Preparation

1. Place your time series data in `data/raw/`
2. Run preprocessing:

```bash
python scripts/preprocess.py \
  --input_dir data/raw/ \
  --output_dir data/processed/ \
  --encoding gramian  # or 'markov', 'recurrence'
```

Expected data shape: `(samples, timesteps, features)`

## Training

### 1. ViT Example

```python
from models.vit import ViT

model = ViT(
    input_shape=(23, 4),
    num_classes=7,
    patch_size=4,
    projection_dim=64
)

history = model.train(
    x_train, y_train,
    x_val, y_val,
    epochs=100,
    batch_size=64
)
```

### 2. Swin Transformer

```python
from models.swin_transformer import SwinTransformer

model = SwinTransformer(
    input_shape=(23, 4),
    num_classes=7,
    window_size=7
)
model.train(...)
```

## Evaluation

```python
metrics = model.evaluate(x_test, y_test)

# Output includes:
# - Accuracy/F1 scores
# - Confusion matrix
# - UMAP visualization
```

## Saved Models

Structure:
```
saved_models/
├── vit/
│   ├── config.json
│   ├── best_model.h5
│   └── metrics.txt
└── swin/
    └── ...
```

To load a pretrained model:
```python
model = ViT.load("saved_models/vit/best_model.h5")
```

## Configuration

Edit `configs/model_configs.yaml`:

```yaml
vit:
  input_shape: [23, 4]
  patch_size: 4
  projection_dim: 64
  num_heads: 4

swin:
  window_size: 7
  shift_size: 3
```



