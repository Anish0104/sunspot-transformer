# Sunspot Transformer

A GPT-style transformer decoder built from scratch in PyTorch, trained to predict monthly sunspot counts from 275 years of solar observation data (1749–present).

---

## What it does

Given the last 128 months of sunspot activity, the model predicts the next month's count — trained autoregressively, the same way GPT models generate text.

**Mean Absolute Error: 16.4 sunspots** on a held-out validation set spanning 1973–2026 (scale: 0–398).

---

## Architecture

Built entirely from scratch — no pre-trained models, no transformer libraries.

```
Input (128 months)
    ↓  Embedding            1 value → 64-dimensional vector
    ↓  Positional Encoding  sine/cosine position fingerprints
    ↓  Self-Attention ×2    4 heads, causal masking, residual + LayerNorm
    ↓  Feedforward ×2       64 → 256 → 64, GELU activation, residual + LayerNorm
    ↓  Output Head          64 → 1 (predicted next month)
```

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feedforward dimension | 256 |
| Context length | 128 months |
| Total parameters | 100,161 |

---

## Data

- **Source:** [SILSO, Royal Observatory of Belgium](https://www.sidc.be/SILSO/monthlyssnplot)
- **Size:** 3,329 monthly observations (Jan 1749 – present)
- **Preprocessing:** Min-max normalization to [0, 1], sliding window of 128 months
- **Split:** 80% train (2,560 sequences), 20% validation (641 sequences)

---

## Results

| Metric | Value |
|---|---|
| Final train loss (MSE) | 0.004139 |
| Final val loss (MSE) | 0.003103 |
| Mean Absolute Error | 16.4 sunspots |
| Validation period | Jan 1973 – May 2026 |

The model correctly learns the shape and periodicity of the ~11-year solar cycle. Train and validation losses stay close throughout training, indicating no overfitting.

---

## Training

```
Epochs:        50
Batch size:    32
Optimizer:     Adam (lr=1e-3)
Loss:          MSE
Hardware:      NVIDIA T4 GPU (Google Colab)
Training time: ~3 minutes
```

---

## Usage

Open `sunspot_transformer.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells. No setup required beyond a GPU runtime.

```
Runtime → Change runtime type → T4 GPU
```

---

## File structure

```
sunspot-transformer/
├── sunspot_transformer.ipynb   # Full notebook: data, model, training, predictions
├── README.md
└── plots/
    ├── training_loss.png        # Train vs val loss curve
    └── predictions.png          # Predicted vs actual sunspot counts
```

---

## Author

**Anish Shirodkar**
M.S. Computer Science (AI & ML), Rutgers University
[Portfolio](https://anishshirodkar.me) · [LinkedIn](https://www.linkedin.com/in/anish-shirodkar/) · [GitHub](https://github.com/Anish0104)
