#!/usr/bin/env python3
"""
Convert metric_definitions.jsonl to Q&A training data for mixing into dataset.

This is Option B from the plan: mix definition examples into training data.
Use this if you want the model to learn definitions during pre-training.

Usage:
    python utils/create_definition_training_data.py

Output:
    - data/metric_definitions_qa.jsonl (for mixing into training)
"""

import json
from pathlib import Path


def create_definition_qa_samples(definitions_path='data/metric_definitions.jsonl',
                                  output_path='data/metric_definitions_qa.jsonl'):
    """
    Convert metric definitions to Q&A format for training.

    Creates multiple question types for each metric:
    - What does [METRIC] measure?
    - What is a bullish signal for [METRIC]?
    - How do you interpret [METRIC] = X?
    """

    # Read definitions
    with open(definitions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    definitions = data['definitions']

    qa_samples = []

    for metric in definitions:
        name = metric['name']
        formula = metric['formula']
        range_val = metric['range']
        bullish = metric['bullish_signal']
        bearish = metric['bearish_signal']
        usage = metric['usage_notes']

        # Question 1: Definition
        qa_samples.append({
            'instruction': f"What does the {name} indicator measure in technical analysis?",
            'output': f"{name} measures: {formula}. Range: {range_val}. {usage}"
        })

        # Question 2: Bullish signal
        qa_samples.append({
            'instruction': f"What is a bullish signal for {name}?",
            'output': bullish
        })

        # Question 3: Bearish signal
        qa_samples.append({
            'instruction': f"What is a bearish signal for {name}?",
            'output': bearish
        })

        # Question 4: Interpretation (if range is numeric)
        if '<' in bullish or '>' in bullish:
            qa_samples.append({
                'instruction': f"How should I interpret {name} values in trading?",
                'output': f"Bullish signals: {bullish}. Bearish signals: {bearish}. {usage}"
            })

    # Save to JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in qa_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Created {len(qa_samples)} Q&A samples from {len(definitions)} metrics")
    print(f"✓ Saved to: {output_path}")
    print(f"\nTo use in training:")
    print("1. Load this file in addition to your main dataset")
    print("2. Mix ~10-20% definition samples with 80-90% task samples")
    print("3. This teaches the model metric meanings during pre-training")
    print(f"\nSample Q&A:")
    print(json.dumps(qa_samples[0], indent=2))

    return qa_samples


def merge_with_training_data(task_data_path, definition_data_path, output_path,
                              definition_ratio=0.15):
    """
    Merge definition Q&A with task training data.

    Args:
        task_data_path: Your main training dataset (JSONL)
        definition_data_path: Output from create_definition_qa_samples()
        output_path: Where to save merged dataset
        definition_ratio: Proportion of definition samples (default: 0.15 = 15%)
    """

    # Load datasets
    with open(task_data_path, 'r', encoding='utf-8') as f:
        task_samples = [json.loads(line) for line in f]

    with open(definition_data_path, 'r', encoding='utf-8') as f:
        definition_samples = [json.loads(line) for line in f]

    # Calculate target counts
    total_task = len(task_samples)
    target_definitions = int(total_task * definition_ratio / (1 - definition_ratio))

    # Sample definitions (with replacement if needed)
    import random
    random.seed(42)
    sampled_definitions = random.choices(definition_samples, k=target_definitions)

    # Merge and shuffle
    merged = task_samples + sampled_definitions
    random.shuffle(merged)

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in merged:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✓ Merged dataset created: {output_path}")
    print(f"  Task samples: {total_task} ({(1-definition_ratio)*100:.0f}%)")
    print(f"  Definition samples: {target_definitions} ({definition_ratio*100:.0f}%)")
    print(f"  Total: {len(merged)} samples")

    return merged


if __name__ == '__main__':
    # Create Q&A samples
    qa_samples = create_definition_qa_samples()

    # Example merge (commented out - user provides their own data path)
    # merged = merge_with_training_data(
    #     task_data_path='data/your_training_data.jsonl',
    #     definition_data_path='data/metric_definitions_qa.jsonl',
    #     output_path='data/merged_training_data.jsonl',
    #     definition_ratio=0.15  # 15% definitions, 85% task
    # )
