#!/usr/bin/env python3
"""
Convert metric_definitions.jsonl to concise reference for system prompt.

Usage:
    python utils/create_metric_reference.py

Output:
    - Prints condensed metric reference to console
    - Saves to data/metric_reference.txt
"""

import json
from pathlib import Path


def create_condensed_reference(definitions_path='data/metric_definitions.jsonl',
                                output_path='data/metric_reference.txt',
                                top_n=15):
    """
    Create concise metric reference from definitions.

    Args:
        definitions_path: Path to metric_definitions.jsonl
        output_path: Where to save condensed reference
        top_n: Number of most important metrics to include (default: 15)
    """

    # Read definitions
    with open(definitions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    definitions = data['definitions']

    # Priority metrics for technical analysis (most commonly used)
    priority_metrics = [
        'RSI', 'MACD', 'MACD_SIGNAL', 'STOCHASTIC_K', 'WILLIAMS_R',
        'SMA_20', 'SMA_50', 'PRICE_TO_SMA20', 'PRICE_TO_SMA50',
        'BB_POSITION', 'ATR_RATIO', 'VOLUME_RATIO', 'RETURN_24H',
        'VOLATILITY_20', 'SMA20_TO_SMA50'
    ]

    # Filter to priority metrics
    priority_defs = [d for d in definitions if d['name'] in priority_metrics]

    # Sort by priority order
    priority_defs.sort(key=lambda d: priority_metrics.index(d['name']))

    # Create condensed reference
    lines = [
        "METRIC REFERENCE (Quick Interpretation Guide)",
        "=" * 60,
        ""
    ]

    for metric in priority_defs:
        # Create concise line: NAME (range): bullish | bearish
        name = metric['name']
        range_val = metric['range']
        bullish = metric['bullish_signal']
        bearish = metric['bearish_signal']

        # Shorten long descriptions
        if len(bullish) > 50:
            bullish = bullish[:47] + "..."
        if len(bearish) > 50:
            bearish = bearish[:47] + "..."

        line = f"• {name} ({range_val})"
        lines.append(line)
        lines.append(f"  ↑ Bullish: {bullish}")
        lines.append(f"  ↓ Bearish: {bearish}")
        lines.append("")

    reference_text = "\n".join(lines)

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(reference_text)

    print(f"✓ Created condensed metric reference: {output_path}")
    print(f"✓ Included {len(priority_defs)} priority metrics\n")
    print("=" * 60)
    print("PREVIEW:")
    print("=" * 60)
    print(reference_text[:1000])
    print(f"\n... ({len(reference_text)} total characters)")
    print("\nTo use in your system prompt:")
    print("1. Copy contents of data/metric_reference.txt")
    print("2. Add to end of your system prompt in config")
    print("3. Or use create_enhanced_system_prompt() below")

    return reference_text


def create_enhanced_system_prompt(base_prompt, metric_reference):
    """
    Combine base system prompt with metric reference.

    Args:
        base_prompt: Your existing system prompt
        metric_reference: Output from create_condensed_reference()

    Returns:
        Enhanced system prompt with metric reference
    """

    enhanced = f"""{base_prompt}

---

{metric_reference}

---

Use the above metric reference to guide your analysis. Focus on the 2-3 most significant indicators for each classification."""

    return enhanced


if __name__ == '__main__':
    # Create condensed reference
    reference = create_condensed_reference()

    # Example: Create enhanced system prompt
    example_base_prompt = """You are a technical analysis expert that evaluates trading indicators and classifies market signals with precision, brevity, and structure.

Your behavior is rule-bound. You must follow all instructions exactly."""

    enhanced = create_enhanced_system_prompt(example_base_prompt, reference)

    print("\n" + "=" * 60)
    print("EXAMPLE ENHANCED SYSTEM PROMPT:")
    print("=" * 60)
    print(enhanced[:500] + "\n...")
