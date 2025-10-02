"""Test brevity bonus in technical analysis reward function."""

from core.custom_rewards import RewardPresetLibrary

# Create the technical analysis reward builder
library = RewardPresetLibrary()
preset = library.get_preset("Technical Analysis Signal")
builder = preset.create_builder()

# Test cases with varying lengths and quality
test_cases = [
    {
        "name": "High quality, very concise (60 words)",
        "instruction": "Analyze: RSI=75, MACD bullish, Price>SMA200",
        "generated": """<analysis>
RSI at 75 indicates overbought conditions which typically signals caution. However, MACD bullish crossover suggests strong upward momentum. Price above SMA200 confirms the overall uptrend remains intact. The combination suggests continuation with some resistance ahead.
</analysis>
<signal>WEAK_BUY</signal>""",
        "reference": """<analysis>
RSI overbought, MACD bullish crossover, price above SMA200 indicates uptrend.
</analysis>
<signal>WEAK_BUY</signal>"""
    },
    {
        "name": "High quality, optimal length (120 words)",
        "generated": """<analysis>
The RSI reading of 75 indicates overbought conditions, which typically suggests the asset may be due for a pullback or consolidation. However, the MACD showing a bullish crossover indicates strong upward momentum and buying pressure in the market. The price trading above the 200-period simple moving average (SMA200) is a strong confirmation that the overall trend remains bullish. When considering all three indicators together, the overbought RSI presents some caution, but the MACD momentum and price position above key support suggest the uptrend has room to continue. This creates a moderately bullish outlook with awareness of potential near-term resistance.
</analysis>
<signal>WEAK_BUY</signal>""",
        "reference": """<analysis>
RSI overbought, MACD bullish crossover, price above SMA200 indicates uptrend.
</analysis>
<signal>WEAK_BUY</signal>"""
    },
    {
        "name": "High quality, very verbose (200+ words)",
        "generated": """<analysis>
The Relative Strength Index (RSI) currently stands at 75, which places the asset firmly in overbought territory. Traditionally, RSI readings above 70 are considered overbought and can signal that the asset is overvalued and may be due for a price correction or consolidation period. This indicator suggests that buying pressure has been extremely strong recently, potentially unsustainably so. However, we must consider this in the broader context of other technical indicators.

The Moving Average Convergence Divergence (MACD) is showing a bullish crossover, which is a positive signal. This occurs when the MACD line crosses above the signal line, indicating increasing upward momentum and suggesting that bullish sentiment is gaining strength in the market. This is typically interpreted as a buy signal by technical analysts.

Furthermore, the current price is trading above the 200-period Simple Moving Average (SMA200), which is one of the most widely watched technical indicators for determining long-term trend direction. When price is above the SMA200, it confirms that the overall trend is bullish and that the asset is in an uptrend.

Taking all three indicators into consideration, while the RSI suggests some caution due to overbought conditions, the MACD momentum and price position above the key moving average support a continuation of the uptrend.
</analysis>
<signal>WEAK_BUY</signal>""",
        "reference": """<analysis>
RSI overbought, MACD bullish crossover, price above SMA200 indicates uptrend.
</analysis>
<signal>WEAK_BUY</signal>"""
    },
    {
        "name": "Low quality, concise (wrong signal)",
        "generated": """<analysis>
RSI is high and MACD shows bullish patterns.
</analysis>
<signal>STRONG_SELL</signal>""",
        "reference": """<analysis>
RSI overbought, MACD bullish crossover, price above SMA200 indicates uptrend.
</analysis>
<signal>WEAK_BUY</signal>"""
    },
    {
        "name": "Medium quality, concise (right direction, wrong strength)",
        "generated": """<analysis>
RSI at 75 shows overbought conditions. MACD bullish crossover indicates momentum. Price above SMA200 confirms uptrend. These factors support a buying opportunity.
</analysis>
<signal>STRONG_BUY</signal>""",
        "reference": """<analysis>
RSI overbought, MACD bullish crossover, price above SMA200 indicates uptrend.
</analysis>
<signal>WEAK_BUY</signal>"""
    }
]

print("=" * 80)
print("TESTING BREVITY BONUS IN TECHNICAL ANALYSIS REWARD")
print("=" * 80)
print()

for i, test_case in enumerate(test_cases, 1):
    print(f"Test Case {i}: {test_case['name']}")
    print("-" * 80)

    generated = test_case['generated']
    reference = test_case['reference']
    instruction = test_case.get('instruction', 'Analyze indicators')

    # Compute reward
    total_reward, components = builder.compute_total_reward(instruction, generated, reference)

    # Calculate word count
    word_count = len(generated.split())

    # Calculate base quality (non-length components)
    quality_components = {k: v for k, v in components.items() if 'length' not in k.lower()}
    base_quality = sum(quality_components.values()) / len(quality_components) if quality_components else 0

    print(f"Word Count: {word_count}")
    print(f"Base Quality (non-length): {base_quality:.3f}")
    print(f"Total Reward: {total_reward:.3f}")
    print()
    print("Component Breakdown:")
    for name, value in components.items():
        print(f"  {name:30s}: {value:.3f}")
    print()

    # Check if brevity bonus was applied
    if 'response_length' in components and base_quality > 0.7:
        if 50 <= word_count <= 150:
            expected_multiplier = 1.0 + (150 - word_count) / 1000 * 1.5
            print(f"✓ Brevity bonus SHOULD apply: {expected_multiplier:.3f}x multiplier")
        else:
            print(f"✓ No brevity bonus (outside 50-150 word range)")
    else:
        if base_quality <= 0.7:
            print(f"✓ No brevity bonus (quality {base_quality:.3f} below 0.7 threshold)")
        else:
            print(f"✓ No length component found")

    print()
    print("=" * 80)
    print()

print("\nKEY INSIGHTS:")
print("-" * 80)
print("1. High quality + concise (60 words) should get brevity bonus (~1.135x)")
print("2. High quality + optimal (120 words) should get smaller bonus (~1.045x)")
print("3. High quality + verbose (200+ words) should get NO bonus (outside range)")
print("4. Low quality + concise should get NO bonus (quality < 0.7)")
print("5. Medium quality (0.7) + concise might get bonus at threshold")
print()
print("This prevents gaming: short-but-wrong gets no advantage!")
