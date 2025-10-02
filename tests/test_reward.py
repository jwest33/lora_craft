"""Test updated reward function for technical analysis."""

from core.custom_rewards import RewardPresetLibrary

# Create library and get preset
lib = RewardPresetLibrary()
preset = lib.get_preset('Technical Analysis Signal')
builder = preset.create_builder()

# Print component weights
print('=' * 80)
print('REWARD COMPONENTS:')
print('=' * 80)
components = builder.get_component_details()
total_weight = sum(c['weight'] for c in components)
print(f'Total weight: {total_weight}')
for c in components:
    print(f"  {c['name']}: {c['weight']} ({c['weight_percentage']:.1f}%)")

# Test cases
print('\n' + '=' * 80)
print('TEST CASES:')
print('=' * 80)

# Good output: proper format, right direction (WEAK vs STRONG), reasonable length
test_good = '<analysis>RSI at 75 indicates overbought. MACD shows bullish crossover above signal line.</analysis><signal>WEAK_BUY</signal>'

# Bad output: rambling, no format, hits token limit
test_bad = 'I think this is a good stock to buy because it has been going up recently and I believe it will continue to rise based on my analysis of the market trends and economic indicators that suggest positive momentum in this sector which makes me confident that this would be a profitable investment opportunity for most investors who are looking to maximize their returns in the current market environment'

# Reference answer
ref = '<analysis>RSI overbought at 75, MACD bullish crossover</analysis><signal>STRONG_BUY</signal>'

# Compute rewards
reward_good, comp_good = builder.compute_total_reward('Analyze', test_good, ref)
reward_bad, comp_bad = builder.compute_total_reward('Analyze', test_bad, ref)

print(f'\nGood output ({len(test_good.split())} words, proper format):')
print(f'  Total reward: {reward_good:.4f}')
for k, v in comp_good.items():
    print(f'    {k}: {v:.3f}')

print(f'\nBad output ({len(test_bad.split())} words, rambling):')
print(f'  Total reward: {reward_bad:.4f}')
for k, v in comp_bad.items():
    print(f'    {k}: {v:.3f}')

print(f'\nReward variance: {abs(reward_good - reward_bad):.4f}')
print('=' * 80)

# Additional test: perfect match
test_perfect = '<analysis>RSI overbought at 75, MACD bullish crossover</analysis><signal>STRONG_BUY</signal>'
reward_perfect, comp_perfect = builder.compute_total_reward('Analyze', test_perfect, ref)

print(f'\nPerfect match ({len(test_perfect.split())} words):')
print(f'  Total reward: {reward_perfect:.4f}')
for k, v in comp_perfect.items():
    print(f'    {k}: {v:.3f}')

print('=' * 80)
print('SUMMARY:')
print(f'  Perfect match: {reward_perfect:.4f}')
print(f'  Good (right direction, wrong strength): {reward_good:.4f}')
print(f'  Bad (no format, rambling): {reward_bad:.4f}')
print(f'  Variance range: {reward_perfect - reward_bad:.4f}')
print('=' * 80)
