# Baseline Loss Experiment - Implementation Guide

## 📋 Quick Reference

### File Structure
```
src/xil_research_modification/experiments/baseline_balanced
├── baseline_loss_experiment.py    # Main implementation
├── baseline_config.yaml            # Configuration file
├── 08MNIST/
│   ├── confounded_v1/train.pth
│   └── original/test.pth
└── model_confounded.pth
```

## 🔧 Implementation Components

### 1. **UserCorrectedDataset**
- Pre-generates N corrected instances by zeroing out pixels marked by binary masks
- **CRITICAL**: Only contains class 8 (confounded class) → creates class imbalance
- Stores both corrected and original inputs for visualization

### 2. **MixedBatchSampler**
- Creates batches with fixed ratio: (batch_size - k) confounded + k corrected
- Default: 60 confounded + 4 corrected per batch of 64
- **TODO**: Future work - implement dynamic ratio schedule (curriculum learning)

### 3. **Loss Functions**

#### BaselineBalancedLoss
```
L = CE(confounded) + λ * CE(corrected)
```
- Explicit weighting of user-corrected samples
- Returns: (total_loss, confounded_loss, corrected_loss) for logging

#### BaselineImbalancedLoss
```
L = CE(full_batch)
```
- Standard cross-entropy, no distinction between sources
- Returns same tuple format (dummy values for consistency)

### 4. **Training Loop**
- `train_one_epoch()`: Single epoch with mixed batches
- `evaluate()`: Standard evaluation on val/test set
- `fit_until_early_stopping()`: Trains until no improvement for N epochs
- All functions log to TensorBoard per epoch

### 5. **TensorBoard Logging**
```
runs/experiment_name/lr0.01_lossbalanced_lambda1.0_pool100/
├── Scalars/
│   ├── Loss/
│   │   ├── train_epoch
│   │   ├── train_confounded
│   │   ├── train_corrected
│   │   └── val_epoch
│   └── Accuracy/
│       ├── train_epoch
│       └── val_epoch
├── Images/
│   ├── confounded_samples
│   ├── corrected_original
│   └── corrected_samples
└── hparams (table comparing all runs)
```

## 🚀 Testing Instructions

### Step 1: Sanity Check (Overfitting Test)
```bash
python baseline_loss_experiment.py \
    --config config_baseline.yaml \
    --experiment overfit_test
```

**Expected behavior:**
- Uses only 100 training samples
- Should overfit quickly (high train accuracy)
- Validates the training loop works correctly

**What to check:**
- Training loss should decrease monotonically
- Accuracy should reach near 100% on training set
- TensorBoard shows images correctly
- No crashes or NaN losses

### Step 2: Find Learning Rate
```bash
python baseline_loss_experiment.py \
    --config config_baseline.yaml \
    --experiment lr_search
```

**Expected behavior:**
- Tests 8 learning rates: [0.1, 0.07, 0.05, 0.01, 0.007, 0.005, 0.001, 0.0001]
- Uses 1000 training samples for faster iteration
- Should identify optimal LR range

**What to check:**
- Too high LR: Loss explodes or oscillates
- Too low LR: Loss decreases very slowly
- Optimal LR: Smooth decrease, good final accuracy
- Use TensorBoard hparams table to compare

### Step 3: Lambda Search (Balanced Loss Only)
```bash
python baseline_loss_experiment.py \
    --config config_baseline.yaml \
    --experiment lambda_search
```

**Expected behavior:**
- Tests λ ∈ [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- Uses best LR from Step 2
- Shows impact of weighting corrected samples

**What to check:**
- λ too small: Corrected samples ignored, no improvement
- λ too large: Model overfits to corrected samples
- Optimal λ: Balance between both loss terms

### Step 4: Full Grid Search
```bash
python baseline_loss_experiment.py \
    --config config_baseline.yaml \
    --experiment full_grid_search
```

**Expected behavior:**
- Uses full training dataset
- Tests best hyperparameters from previous steps
- Compares Balanced vs Imbalanced loss

**What to check:**
- Does Balanced loss outperform Imbalanced?
- How much does pool size matter?
- Final test accuracy on original MNIST

## 🐛 Debugging Checklist

### If training loss is NaN:
1. Check learning rate (try smaller)
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
3. Verify inputs are normalized correctly
4. Check for division by zero in loss computation

### If accuracy doesn't improve:
1. Verify data loading: print batch contents
2. Check labels: are they one-hot encoded correctly?
3. Verify mask creation: `is_user_corrected` should be True for last k samples
4. Check if model is actually updating: print parameter norms

### If memory issues:
1. Reduce batch size
2. Reduce user_corrected_pool_size
3. Clear CUDA cache: `torch.cuda.empty_cache()`

### If TensorBoard images are wrong:
1. Check normalization in `vutils.make_grid(normalize=True)`
2. Verify indices: are we sampling class 8 correctly?
3. Check mask application: corrected images should have no dots

## 📊 Expected Results

### Hypothesis:
- **BaselineBalancedLoss** should outperform **BaselineImbalancedLoss**
- Why? Explicit weighting helps model focus on corrected samples
- The model should learn to ignore spurious dots and focus on digit shape

### Metrics to watch:
1. **Validation accuracy**: Should improve from baseline (~60% on confounded model)
2. **Test accuracy**: Final performance on original MNIST
3. **Loss breakdown**: Is corrected loss decreasing?

### If results are unexpected:
- Check if pool size is too small (not enough corrected examples)
- Check if λ is mistuned (corrected samples over/under-weighted)
- Verify early stopping isn't too aggressive (might need more epochs)

## 🔍 Code Review Points

### Critical sections to verify:
1. **MixedBatchSampler ordering**: Last k samples must be corrected
2. **Mask creation**: `create_is_user_corrected_mask()` must match sampler
3. **Loss computation**: Balanced loss should compute separate terms
4. **One-hot to indices**: `targets.argmax(dim=-1)` for CrossEntropyLoss

### Potential issues from old code:
- ✅ Fixed: Custom batch sampler is simpler (no origin indices tracking)
- ✅ Fixed: Loss functions return consistent tuple format
- ✅ Fixed: Clear separation of train/val/test evaluation
- ✅ Fixed: TensorBoard logging is per-epoch (not per-batch)

## 📝 Next Steps After Baseline Results

1. **Analyze results**: Which loss works better? Why?
2. **Visualize attention**: Use GradCAM on corrected samples
3. **Implement Lagrangian loss**: Port from old code or redesign?
4. **Compare all three approaches**: Balanced vs Imbalanced vs Lagrangian

## 🤔 Questions to Answer with Experiments

1. Does explicit weighting (Balanced) help more than implicit (Imbalanced)?
2. How many corrected samples do we need? (pool size sensitivity)
3. What's the optimal λ? (trade-off between confounded and corrected)
4. Does the model still rely on dots after correction? (check with GradCAM)
5. How does this compare to Lagrangian constraint optimization?

---

## 🎯 Summary

**Goal**: Correct a CNN trained on confounded MNIST data (8s with spurious dots)

**Method**: 
- Mix confounded training data with user-corrected samples (dots removed)
- Compare two loss functions:
  1. Balanced: Explicitly weights corrected samples
  2. Imbalanced: Treats all samples equally

**Experiment design**:
- Pre-generate N corrected instances from training set
- Each batch: 60 confounded + 4 corrected
- Train with early stopping (validation-based)
- Grid search over: LR, loss type, λ, pool size

**Expected outcome**: 
- Model learns to ignore spurious correlation
- Validation/test accuracy improves on original MNIST
- Establishes baseline for future Lagrangian optimization

**Next steps**: 
- Implement Lagrangian constraint optimization
- Compare all three approaches
- Analyze which method best removes spurious correlation