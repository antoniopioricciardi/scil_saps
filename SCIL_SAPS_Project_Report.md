# Project Report: SCIL + SAPS for Zero-Shot Policy Stitching
## Experimental Validation on Super Mario Bros

**Author**: Antonio Ricciardi
**Date**: December 2024
**Context**: Validating SAPS and supervised contrastive learning for transfer to online RL with R3L

---

## Executive Summary

This project successfully validated **SAPS (Semantic Alignment via Prototypical Subspace)** for zero-shot policy stitching using **SCIL (Supervised Contrastive Imitation Learning)** on Super Mario Bros. The key finding: **action-conditioned embeddings enable near-perfect cross-model transfer with minimal alignment data**, achieving **100.1% of native performance** after stitching.

**Critical Insight for RL**: Supervised contrastive loss creates semantically meaningful latent spaces where action labels serve as natural anchors for alignment—directly applicable to R3L's anchor-based framework.

---

## What We Did

### 1. Training: SCIL Encoders with Action-Conditioned Representations

**Setup**:
- Trained **EfficientNet-B1** encoders on expert demonstrations from two Mario levels (1-1 and 1-2)
- Used supervised contrastive loss to cluster embeddings by action labels
- Joint objective: `L_total = L_task + λ·L_SupCon(h, actions)`
- Configuration: λ=1.0, temperature=0.05, batch size=32, 80 epochs

**Results**:
- Level 1-1: **93.43%** validation accuracy
- Level 1-2: **94.51%** validation accuracy
- Latent space analysis shows **excellent action-based clustering**:
  - Silhouette score: **0.53** (good separation)
  - k-NN accuracy: **98.5%** (excellent separability)
  - Separation ratio: **2.98** (inter-class distance 3× intra-class)

### 2. Alignment: SAPS Transformation Estimation

**Method**:
- Collected **357 anchor pairs** using action label matching
- Estimated affine transformation `τ(h) = Rh + b` via SVD
- Alignment took **<1 minute** on CPU

**Quality**:
- Cosine similarity between matched anchors:
  - Before alignment: **~0.45** (misaligned spaces)
  - After alignment: **~0.92** (near-perfect alignment)
- t-SNE visualization confirms clusters align across models

### 3. Evaluation: Zero-Shot Policy Stitching

**Test**: Encoder from Level 1-1 + Policy Head from Level 1-2 (via SAPS transformation)

**Results**:
| Configuration | Accuracy |
|--------------|----------|
| Native Model 1 (1-1 encoder + 1-1 policy) | 98.05% |
| Native Model 2 (1-2 encoder + 1-2 policy) | 98.24% |
| **Stitched** (1-1 encoder + transform + 1-2 policy) | **98.24%** |

**Performance**: **100.1% of native performance** with zero retraining

---

## Key Findings for R3L & Online RL

### 1. Action Labels as Semantic Anchors

**Observation**: SCIL's action-conditioned embeddings naturally provide semantic alignment points.

**Implication for R3L**:
- Use action labels to select anchors instead of random sampling
- Anchor sets should contain diverse action representations
- Action clustering can automatically identify high-quality anchor candidates

**Recommendation**: Implement action-stratified anchor sampling in R3L:
```python
# Instead of random anchors
anchors = buffer.sample(N)

# Use action-stratified sampling
anchors = []
for action in unique_actions:
    anchors.append(buffer.sample(N // num_actions, action=action))
```

### 2. Supervised Contrastive Loss Improves Alignment Quality

**Observation**: Embeddings trained with SupCon loss showed:
- Tighter intra-action clusters (std ~0.9)
- Better inter-action separation (3× ratio)
- Near-perfect SAPS alignment (0.92 cosine similarity)

**Implication for R3L**:
- Adding SupCon loss during RL training can stabilize relative representations
- Action-based clustering complements anchor-based encoding
- May reduce sensitivity to anchor set selection

**Recommendation**: Augment R3L training objective:
```python
L_total = L_RL(π, V) + λ_R3L·L_relative(z, A) + λ_SupCon·L_SupCon(h, actions)
```

### 3. Minimal Data Requirements for Post-Training Alignment

**Observation**: Only **357 anchor pairs** (~5-10 minutes of gameplay) achieved near-perfect alignment.

**Implication for R3L**:
- SAPS can retrofit existing RL models without retraining
- Useful for aligning models trained before R3L implementation
- Enables hybrid deployment: R3L for new models, SAPS for legacy models

**Use Case**: Aligning pre-trained foundation models (CLIP, DINOv2) with RL policies:
1. Collect ~1000 observations with action labels
2. Estimate SAPS transformation (1 minute)
3. Deploy aligned encoder with existing policy

### 4. Cross-Level Transfer Validates Cross-Domain Potential

**Observation**: Encoders trained on visually different levels (1-1 vs 1-2) transferred seamlessly.

**Implication for R3L**:
- Validates potential for sim-to-real transfer
- Suggests action semantics are domain-invariant
- Reinforces multi-hop composition viability (sim → toy robot → real robot)

---

## Practical Integration Strategy

### For New RL Training Runs

**Use R3L + SupCon**:
```python
# 1. Add SupCon loss to R3L training
L_total = L_RL + λ_R3L·L_relative + λ_SupCon·L_SupCon

# 2. Use action-stratified anchor sampling
anchors = select_action_stratified_anchors(buffer, num_anchors=32)

# 3. Compute relative representations
z = sim(φ(o), φ(anchors))
```

### For Existing Pre-Trained Models

**Use SAPS Post-Alignment**:
```python
# 1. Collect anchor pairs with action labels
anchors = collect_matched_observations(env, num_samples=1000)

# 2. Estimate transformation (fast)
R, b = estimate_saps_transform(encoder_1(anchors), encoder_2(anchors))

# 3. Deploy stitched policy
def stitched_policy(obs):
    h = encoder_1(obs)
    h_aligned = h @ R + b
    return policy_2(h_aligned)
```

### Hybrid Approach (Recommended)

1. Train encoders with **R3L + SupCon** (best latent structure)
2. Use **action-stratified anchors** (automatic semantic selection)
3. Keep **SAPS as fallback** for retrofitting legacy models
4. Update transformations **online** as environment drifts

---

## Experimental Artifacts

### Models
- `checkpoints/scil_encoder_mario_1_1_efficientnet_b1_lam1.pth` (Level 1-1)
- `checkpoints/scil_encoder_mario_1_2_efficientnet_b1_lam1.pth` (Level 1-2)
- `notebooks/scil_stitched_1_1_enc_to_1_2_pol.pth` (Stitched model)

### Visualizations
- `notebooks/alignment_quality.png` - Cosine similarity before/after alignment
- `notebooks/latent_space_scil.png` - t-SNE visualization of action clusters
- `notebooks/confusion_matrix_scil.png` - k-NN classification performance
- `notebooks/saps_alignment_animation.gif` - Transformation visualization

### Code
- `train_scil_pretrained.py` - Training script with SupCon loss
- `notebooks/semantic_alignment.ipynb` - Full SAPS implementation
- `notebooks/latent_analysis.ipynb` - Clustering quality analysis

---

## Recommendations for R3L Implementation

### Immediate Actions

1. **Add action-conditioned anchor sampling** to R3L training
   - Maintain anchor diversity across action space
   - Update anchors using action-stratified EMA

2. **Integrate supervised contrastive loss** as auxiliary objective
   - Start with λ_SupCon = 1.0, temperature = 0.05
   - Monitor silhouette score to validate clustering

3. **Implement SAPS fallback** for existing models
   - Estimate transformations with ~1000 samples
   - Use for sim-to-real with minimal real-world data

### Research Questions to Explore

1. **Does SupCon + R3L improve anchor stability?**
   - Hypothesis: Action clustering reduces anchor drift during online training
   - Test: Compare anchor set variance with/without SupCon

2. **Can action-stratified anchors reduce anchor set size?**
   - Hypothesis: Semantic diversity matters more than quantity
   - Test: Compare 32 stratified vs 64 random anchors

3. **How does SAPS + R3L perform in sim-to-real?**
   - Scenario: Train sim encoder with R3L, align to real policy via SAPS
   - Test: <1000 real samples sufficient for deployment?

4. **Multi-hop composition quality bounds?**
   - Chain: Sim-R3L → SAPS(toy robot) → SAPS(real robot)
   - Measure: Alignment quality degradation across hops

---

## Conclusion

This project validated that:

1. **SAPS works exceptionally well** for zero-shot policy stitching (100.1% native performance)
2. **Action-conditioned embeddings** provide natural semantic anchors
3. **Supervised contrastive loss** creates alignment-friendly latent spaces
4. **Minimal data** (<1000 samples) enables post-training alignment

**For your R3L project**: Integrate action-stratified anchors and SupCon loss during training, use SAPS for retrofitting existing models. The combination provides flexibility for any deployment scenario while maintaining strong alignment quality.

---

## References

- **SAPS**: [Mapping representations in Reinforcement Learning via Semantic Alignment for Zero-Shot Stitching](https://arxiv.org/html/2503.01881)
- **SCIL**: [Learning Representations in Video Game Agents with Supervised Contrastive Imitation Learning](https://arxiv.org/html/2509.11880)
- **R3L**: Ricciardi, A.P., et al. R3L: Relative Representations for Reinforcement Learning

---

**Next Steps**: Apply these insights to online RL experiments with R3L on continuous control tasks (MuJoCo, Isaac Gym) to validate sim-to-real transfer capabilities.
