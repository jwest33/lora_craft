# Using Metric Definitions for Training

This guide explains how to use your `data/metric_definitions.jsonl` file to improve model understanding of financial metrics.

## ⚠️ Don't Do 2-Step Pre-Training

**Recommendation:** Do NOT do separate pre-training on definitions, then task data. This causes:
- ❌ Catastrophic forgetting (Step 2 overwrites Step 1)
- ❌ 2x training time
- ❌ Format consistency issues
- ❌ Unnecessary complexity

## ✅ Recommended Approaches

### **Option A: Augmented System Prompt** (BEST - Start Here)

Add metric reference directly to your system prompt. Zero training overhead, always available to model.

**Steps:**

1. Run the utility:
```bash
python utils/create_metric_reference.py
```

2. This creates `data/metric_reference.txt` with condensed metric guide

3. Copy contents and add to your config's `system_prompt`:
```json
{
  "template": {
    "system_prompt": "You are a technical analysis expert...\\n\\n[paste metric reference here]"
  }
}
```

**Pros:**
- ✅ Zero training time
- ✅ Always available
- ✅ Easy to update
- ✅ No forgetting risk

**When to use:** Try this first! See if adding metric reference improves reasoning quality.

---

### **Option B: Mixed Training Dataset** (Alternative)

Mix metric Q&A into your training data during pre-fine-tuning.

**Steps:**

1. Create Q&A samples from definitions:
```bash
python utils/create_definition_training_data.py
```

2. This creates `data/metric_definitions_qa.jsonl` with ~150 Q&A samples

3. **Option B1 - Load both datasets separately:**
   - Upload both files to the UI
   - Configure to use both in training
   - Model sees task + definition examples

4. **Option B2 - Merge into single file:**
```python
python utils/create_definition_training_data.py
# Edit the script to uncomment merge_with_training_data() at the bottom
# Provide path to your task data
# Creates merged dataset with 85% task, 15% definitions
```

**Pros:**
- ✅ Model practices using definitions
- ✅ Natural curriculum (mixes both types)
- ✅ Single training run

**Cons:**
- ⚠️ Adds training samples (longer training)
- ⚠️ Need to balance ratio (too many definitions → less task practice)

**When to use:** If Option A doesn't improve quality enough, try this.

---

## 📊 Comparison

| Approach | Training Time | Complexity | Risk | Effectiveness |
|----------|--------------|------------|------|---------------|
| **A: Augmented Prompt** | None | Low | None | Medium-High |
| **B: Mixed Dataset** | +15% | Medium | Low | High |
| ❌ 2-Step Pre-training | 2x | High | High | Unknown |

## 🎯 Recommended Workflow

1. **Start with Option A** (augmented system prompt)
   - Run `create_metric_reference.py`
   - Add to your system prompt
   - Test if quality improves

2. **If needed, try Option B** (mixed dataset)
   - Run `create_definition_training_data.py`
   - Mix 10-20% definitions into training
   - Compare quality vs Option A

3. **Never use 2-step pre-training** for this use case
   - Risk/reward is poor
   - Better alternatives exist

## 💡 Pro Tips

1. **Start simple:** Option A is often enough!

2. **Test incrementally:**
   - Baseline: No definitions
   - Test 1: Add metric reference to prompt
   - Test 2: Mix definitions into training
   - Compare which works best

3. **Monitor for degradation:**
   - Adding too much to system prompt can confuse model
   - Keep metric reference concise (<1000 chars)

4. **Combine approaches:**
   - Short reference in system prompt (top 5 metrics)
   - Mixed Q&A in training data (all metrics)
   - Best of both worlds!

## 📝 Examples

### Option A: Enhanced System Prompt

```
You are a technical analysis expert that evaluates trading indicators.

METRIC REFERENCE:
• RSI (0-100): Bullish if <30 (oversold) | Bearish if >70 (overbought)
• MACD: Bullish if > MACD_SIGNAL | Bearish if < MACD_SIGNAL
• PRICE_TO_SMA20 (0.9-1.1): Bullish if >1.0 | Bearish if <1.0
[... 12 more key metrics]

Focus on the 2-3 most significant indicators for classification.
```

### Option B: Mixed Training Sample

```json
{
  "instruction": "What is a bullish signal for RSI?",
  "output": "RSI < 30 indicates oversold conditions, which is a bullish signal suggesting potential buying opportunity."
}
```

## 🔧 Troubleshooting

**Q: Model still doesn't understand metrics well?**
- Try Option B (mixed training)
- Increase definition ratio to 20-30%
- Ensure definitions use same format as task data

**Q: Quality got worse after adding metric reference?**
- Reference might be too long - use top 10 metrics only
- Ensure reference doesn't conflict with existing instructions
- Try simpler format

**Q: Should I do 2-step pre-training anyway?**
- No! Use continual learning techniques if you must
- Or try Option B which achieves similar goals safely
