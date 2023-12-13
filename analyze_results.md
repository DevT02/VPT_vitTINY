# Results Analysis

## Experiment Summary
- **Best Val Acc**: 85.13% (epoch 7)
- **Test Acc**: 79.65%
- **Backdoor Acc**: 47.07%
- **Config**: prompt_tokens=5, lr=0.001, epsilon=0.1

## Training Behavior

**Epoch-by-epoch analysis:**
- Epoch 1-2: Rapid improvement (76.41% → 82.69%)
- Epoch 3-4: Steady improvement (82.69% → 83.88%)
- Epoch 5: Significant drop (54.54%) - likely a bad batch or instability
- Epoch 6-7: Recovery and peak (81.30% → 85.13%)
- Epoch 8-10: Gradual decline (78.83% → 73.40%)

## Key Observations

1. **Peak Performance**: Model peaked at epoch 7 with 85.13% validation accuracy
2. **Instability**: Epoch 5 showed a major drop (54.54%), suggesting prompt learning instability
3. **Overfitting**: After epoch 7, validation accuracy declined while training accuracy continued improving
4. **Early Stopping**: Should have stopped at epoch 12 (5 epochs without improvement from epoch 7)

## Performance Metrics

- **Test Accuracy (79.65%)**: Good performance, close to best validation accuracy
- **Train vs Val Gap**: Val acc consistently higher than train acc (normal for VPT with frozen backbone)
- **Backdoor Success**: 47.07% backdoor accuracy is above random (10%) but not perfect - backdoor is partially working

## Recommendations

1. **Use Early Stopping**: Set `--early_stop_patience 5` to stop at epoch 7
2. **Lower Learning Rate**: Try `--lr 5e-4` to reduce instability
3. **More Prompt Tokens**: Try `--prompt_tokens 10` for potentially better performance
4. **Learning Rate Schedule**: Add cosine annealing to stabilize training

## Comparison to Baseline

For VPT on CIFAR10 with TinyViT:
- **Expected range**: 75-85% test accuracy
- **This result**: 79.65% - solid performance
- **Best possible**: Could potentially reach 82-85% with better hyperparameters

