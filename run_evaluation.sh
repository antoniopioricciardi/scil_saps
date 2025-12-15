#!/bin/bash
# Complete evaluation workflow for SAPS experiments

echo "================================================================"
echo "SAPS EVALUATION WORKFLOW"
echo "================================================================"

# Configuration
EPISODES=20
LEVEL_1="1-1"
LEVEL_2="1-2"
MODEL_1="scil_encoder_mario_1_1_efficientnet_b1_lam2.pth"
MODEL_2="scil_encoder_mario_1_2_efficientnet_b1_lam2.pth"
STITCHED="scil_stitched_1_1_enc_to_1_2_pol.pth"

echo ""
echo "Configuration:"
echo "  Episodes per test: $EPISODES"
echo "  Level 1: $LEVEL_1"
echo "  Level 2: $LEVEL_2"
echo "  Model 1: $MODEL_1"
echo "  Model 2: $MODEL_2"
echo "  Stitched: $STITCHED"
echo ""

# Create results directory
mkdir -p results
cd results || exit

echo "================================================================"
echo "1. BASELINE: Native models on their own levels"
echo "================================================================"

echo ""
echo "[1/4] Testing Model 1 on Level $LEVEL_1 (native)..."
python ../test_mario_agent.py \
    --model "../$MODEL_1" \
    --model-type native \
    --level "$LEVEL_1" \
    --episodes $EPISODES \
    --output "native_1_on_1.json"

echo ""
echo "[2/4] Testing Model 2 on Level $LEVEL_2 (native)..."
python ../test_mario_agent.py \
    --model "../$MODEL_2" \
    --model-type native \
    --level "$LEVEL_2" \
    --episodes $EPISODES \
    --output "native_2_on_2.json"

echo ""
echo "================================================================"
echo "2. CROSS-LEVEL: Native models on different levels"
echo "================================================================"

echo ""
echo "[3/4] Testing Model 1 on Level $LEVEL_2 (cross-level)..."
python ../test_mario_agent.py \
    --model "../$MODEL_1" \
    --model-type native \
    --level "$LEVEL_2" \
    --episodes $EPISODES \
    --output "native_1_on_2.json"

echo ""
echo "[4/4] Testing Model 2 on Level $LEVEL_1 (cross-level)..."
python ../test_mario_agent.py \
    --model "../$MODEL_2" \
    --model-type native \
    --level "$LEVEL_1" \
    --episodes $EPISODES \
    --output "native_2_on_1.json"

echo ""
echo "================================================================"
echo "3. SAPS: Stitched model evaluation"
echo "================================================================"

if [ -f "../$STITCHED" ]; then
    echo ""
    echo "[5/6] Testing Stitched Model (Enc 1 + Pol 2) on Level $LEVEL_1..."
    python ../test_mario_agent.py \
        --model "../$STITCHED" \
        --model-type stitched \
        --encoder-path "../$MODEL_1" \
        --policy-path "../$MODEL_2" \
        --level "$LEVEL_1" \
        --episodes $EPISODES \
        --output "stitched_enc1_pol2_on_1.json"

    echo ""
    echo "[6/6] Testing Stitched Model (Enc 1 + Pol 2) on Level $LEVEL_2..."
    python ../test_mario_agent.py \
        --model "../$STITCHED" \
        --model-type stitched \
        --encoder-path "../$MODEL_1" \
        --policy-path "../$MODEL_2" \
        --level "$LEVEL_2" \
        --episodes $EPISODES \
        --output "stitched_enc1_pol2_on_2.json"
else
    echo ""
    echo "WARNING: Stitched model not found: $STITCHED"
    echo "Run the notebook to create stitched model first!"
fi

echo ""
echo "================================================================"
echo "4. COMPARISON & ANALYSIS"
echo "================================================================"

echo ""
echo "Comparing all results..."
python ../compare_results.py *.json --output comparison_all.png

echo ""
echo "================================================================"
echo "EVALUATION COMPLETE!"
echo "================================================================"
echo ""
echo "Results saved in: results/"
echo ""
echo "Generated files:"
ls -lh *.json *.png 2>/dev/null || echo "  (no files generated)"
echo ""
echo "Next steps:"
echo "  1. Review comparison_all.png for visual comparison"
echo "  2. Check individual JSON files for detailed statistics"
echo "  3. Run with --render flag to watch agents play"
echo ""

cd ..
