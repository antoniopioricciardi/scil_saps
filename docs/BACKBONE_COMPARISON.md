# Pretrained Backbone Comparison

## How to Choose:

Simply edit `train_scil_pretrained.py` line 18:

```python
BACKBONE = "resnet18"           # Default: Fast, proven
BACKBONE = "efficientnet-b0"    # Efficient: Better accuracy/compute ratio
BACKBONE = "efficientnet-b1"    # Balanced: More capacity
BACKBONE = "efficientnet-b2"    # Powerful: Best accuracy (slower)
```

## Comparison:

| Backbone | Parameters | Speed | Accuracy | Best For |
|----------|-----------|-------|----------|----------|
| **ResNet18** | 11.7M | ⚡⚡⚡ Fast | Good | Quick experiments, baseline |
| **EfficientNet-B0** | 5.3M | ⚡⚡⚡ Fast | Better | Resource-constrained, mobile |
| **EfficientNet-B1** | 7.8M | ⚡⚡ Medium | Better+ | Balanced performance |
| **EfficientNet-B2** | 9.2M | ⚡ Slower | Best | Maximum accuracy |

## Key Differences:

### **ResNet18:**
- ✅ Well-studied, reliable
- ✅ Fast training
- ⚠️ Older architecture (2015)
- ⚠️ Less parameter-efficient

### **EfficientNet:**
- ✅ Modern architecture (2019)
- ✅ Better accuracy per parameter
- ✅ Compound scaling (width/depth/resolution)
- ✅ Better ImageNet performance
- ⚠️ Slightly more complex

## Why Try EfficientNet for Mario?

1. **Better feature quality:** EfficientNet-B0 achieves ~77% ImageNet top-1 vs ResNet18's ~70%
2. **More efficient:** Fewer parameters but better representations
3. **Compound scaling:** Balanced scaling of network dimensions
4. **Modern design:** Incorporates squeeze-and-excitation, optimized for transfer learning

## Expected Results:

For Mario with proper data:

| Metric | ResNet18 | EfficientNet-B0 | EfficientNet-B1 |
|--------|----------|-----------------|-----------------|
| Validation Acc | ~88-92% | ~89-93% | ~90-94% |
| Training Speed | 1.0x | 1.0x | 0.8x |
| GPU Memory | Baseline | Similar | +10% |
| Feature Quality | Good | Better | Better+ |

## Recommendation:

**Start with EfficientNet-B0:**
- Similar speed to ResNet18
- Better feature quality
- May give better latent space separation
- If memory allows, try B1 for even better results

## After Training:

Models save as:
- `scil_encoder_mario_resnet18.pth`
- `scil_encoder_mario_efficientnet_b0.pth`
- `scil_encoder_mario_efficientnet_b1.pth`

Compare them in `latent_analysis.ipynb` to see which gives cleaner action clustering!
