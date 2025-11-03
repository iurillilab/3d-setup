# Research Notes: Temporal DAE for Keypoint Denoising

## Current Status
**Temporal Transformer (w=2)**: Test loss ~9-10 (85% improvement over standard DAE ~50-60)

---

## 1. Hyperparameter Cross-Validation

### Run CV Search
```bash
cd threed_utils/denoising
conda activate 3d_setup
python cv_search.py /path/to/data.npy
```

### Grid Parameters
- `latent_dim`: [64, 128, 256]
- `window_size`: [1, 2, 3]  
- `lr`: [1e-3, 5e-4]
- 5-fold CV, 50 epochs per fold
- **Total**: 18 configs × 5 folds = 90 runs

### Output
- `cv_results.json`: All results sorted by mean validation loss
- Best config will be printed

---

## 2. Realistic Masking Strategies

### Current: Zero Masking
Sets masked keypoints to 0.0 (unrealistic but works)

### New Strategies

#### A. **Gaussian Noise**
```bash
python train_temporal.py --data_path /path/to/data.npy \
  --mask_strategy gaussian --noise_level 0.05 \
  --epochs 100 --window_size 2
```
- Adds Gaussian noise (5% of pose scale)
- Simulates tracking jitter/uncertainty
- **Use case**: Continuous tracking errors

#### B. **Dropout**  
```bash
python train_temporal.py --data_path /path/to/data.npy \
  --mask_strategy dropout \
  --epochs 100 --window_size 2
```
- Randomly drops 1-2 coordinates per keypoint
- Simulates partial detection failures
- **Use case**: Occlusions, lost markers

#### C. **Swap**
```bash
python train_temporal.py --data_path /path/to/data.npy \
  --mask_strategy swap \
  --epochs 100 --window_size 2
```
- Swaps left↔right keypoints (ear_lf↔ear_rt, etc.)
- Simulates identity confusion
- **Use case**: Symmetric animal poses

#### D. **Mixed**
```bash
python train_temporal.py --data_path /path/to/data.npy \
  --mask_strategy mixed \
  --epochs 100 --window_size 2
```
- Randomly applies zero/gaussian/dropout
- Most diverse augmentation
- **Use case**: General robustness

---

## 3. Experiments to Run

### A. Best Hyperparameters (after CV)
```bash
# Use best config from cv_results.json
python train_temporal.py --data_path /path/to/data.npy \
  --latent_dim BEST_H --window_size BEST_W --lr BEST_LR \
  --epochs 200 --log_dir logs/best_config
```

### B. Realistic Masking Comparison
```bash
# Zero (baseline)
python train_temporal.py --mask_strategy zero --log_dir logs/mask_zero

# Gaussian
python train_temporal.py --mask_strategy gaussian --log_dir logs/mask_gaussian

# Dropout  
python train_temporal.py --mask_strategy dropout --log_dir logs/mask_dropout

# Mixed (best?)
python train_temporal.py --mask_strategy mixed --log_dir logs/mask_mixed
```

Compare in TensorBoard:
```bash
tensorboard --logdir logs
```

---

## 4. Expected Results

### Hypothesis 1: Hyperparameters
- **Window size**: w=2 likely optimal (balance context/overfitting)
- **Latent dim**: 128 or 256 (not 64, likely underfits)
- **LR**: 1e-3 fine for AdamW + scheduler

### Hypothesis 2: Masking Strategy
- **Zero**: Easiest to learn (clear signal)
- **Gaussian**: Slightly harder, but more robust to jitter
- **Dropout**: Harder, models partial failures
- **Mixed**: Most general, best for real deployment

**Prediction**: Mixed or Gaussian will generalize best to real tracking noise

---

## 5. Analysis Plan

After experiments complete:

1. **Extract results**:
```python
import json
with open('cv_results.json') as f:
    results = json.load(f)
best = results[0]  # Sorted by loss
print(f"Best config: {best['config']}")
print(f"Loss: {best['mean_val_loss']:.2f} ± {best['std_val_loss']:.2f}")
```

2. **Compare masking strategies in TensorBoard**:
   - Train/test loss curves
   - Per-keypoint errors
   - 3D visualizations

3. **Statistical test**:
   - Paired t-test between masking strategies
   - Check if improvement is significant

---

## 6. Next Steps (After Results)

If results show improvement:
1. Train final model with best config + best masking
2. Save checkpoint
3. Implement inference script for real data
4. Apply to full tracking pipeline

If no improvement:
1. Check if model is actually learning (loss decreasing?)
2. Verify masking is applied correctly (visualize)
3. Try simpler strategies first
4. May need more data or different architecture

---

## Notes

- All experiments use same data split (seed=42)
- Logs saved separately for each run
- CV uses 5-fold, final training uses single 90/10 split
- Visualizations logged every 10 epochs


