"""Test script for reward-integrated testing functionality.

This script demonstrates:
1. Loading reward config from a session
2. Single test with reward evaluation
3. Batch test with reward scoring

Usage:
    python tests/test_reward_integration.py --session-id <session_id>
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_tester import ModelTester, TestConfig
from core.batch_tester import BatchTestRunner
from core.session_registry import SessionRegistry
from core.reward_loader import load_reward_from_session, get_reward_component_details


def test_load_reward_config(session_id: str):
    """Test loading reward configuration from session."""
    print("=" * 80)
    print("TEST 1: Loading Reward Configuration from Session")
    print("=" * 80)

    # Load session
    registry = SessionRegistry()
    session_info = registry.get_session(session_id)

    if not session_info:
        print(f"❌ Session '{session_id}' not found")
        return False

    print(f"✓ Session found: {session_info.display_name or session_id}")

    # Load reward function
    reward_builder = load_reward_from_session(session_info)

    if not reward_builder:
        print("❌ Could not load reward configuration from session")
        return False

    print(f"✓ Loaded reward function with {len(reward_builder.rewards)} components")

    # Get component details
    component_details = get_reward_component_details(session_info)

    if component_details:
        print("\nReward Components:")
        for comp in component_details:
            print(f"  - {comp['name']}: weight={comp['weight']:.2f} ({comp['weight_percentage']:.1f}%)")
            print(f"    Type: {comp['type']}, Class: {comp['class']}")
            if comp['parameters']:
                print(f"    Parameters: {comp['parameters']}")

    print("\n✓ TEST 1 PASSED\n")
    return True


def test_single_test_with_reward(session_id: str):
    """Test single model generation with reward evaluation."""
    print("=" * 80)
    print("TEST 2: Single Test with Reward Evaluation")
    print("=" * 80)

    # Load session
    registry = SessionRegistry()
    session_info = registry.get_session(session_id)

    if not session_info or not session_info.checkpoint_path:
        print("❌ Session not found or no checkpoint available")
        return False

    # Create model tester
    tester = ModelTester()

    # Load model
    print(f"Loading model from: {session_info.checkpoint_path}")
    success, error = tester.load_trained_model(session_info.checkpoint_path, session_id)

    if not success:
        print(f"❌ Failed to load model: {error}")
        return False

    print("✓ Model loaded successfully")

    # Test prompt (technical analysis example)
    test_prompt = """Based on the following technical indicators, provide your analysis and signal:
RSI: 68
MACD: 2.5 (above signal line)
MACD_SIGNAL: 1.8
Price above SMA_20 and SMA_50
Volume increasing"""

    reference_response = """<analysis>
RSI: 68 indicates approaching overbought conditions but not extreme
MACD: Bullish crossover with MACD above signal line at 2.5 vs 1.8
Price structure: Trading above both key moving averages confirms uptrend
Volume: Increasing volume supports the move
</analysis>
<signal>
WEAK_BUY
</signal>"""

    print("\nGenerating response with reward evaluation...")

    # Test with reward evaluation
    result = tester.test_with_reward(
        prompt=test_prompt,
        model_type="trained",
        model_key=session_id,
        session_info=session_info,
        reference_response=reference_response,
        config=TestConfig(temperature=0.7, max_new_tokens=512)
    )

    if not result.get('success'):
        print(f"❌ Generation failed: {result.get('error')}")
        return False

    print("✓ Generation successful")
    print(f"\nResponse:\n{result['response'][:200]}...")

    # Check reward evaluation
    reward_eval = result.get('reward_evaluation')
    if reward_eval and reward_eval.get('success'):
        print(f"\n✓ Reward Evaluation:")
        print(f"  Total Reward: {reward_eval['total_reward']:.3f}")
        print(f"  Components:")
        for comp_name, comp_score in reward_eval['components'].items():
            print(f"    - {comp_name}: {comp_score:.3f}")
    else:
        print(f"❌ Reward evaluation failed: {reward_eval.get('error') if reward_eval else 'Not evaluated'}")

    print("\n✓ TEST 2 PASSED\n")
    return True


def test_batch_test_with_rewards(session_id: str, test_file: str = None):
    """Test batch testing with reward scoring."""
    print("=" * 80)
    print("TEST 3: Batch Test with Reward Scoring")
    print("=" * 80)

    # Create test dataset if not provided
    if not test_file:
        test_file = "./tests/test_prompts.csv"
        print(f"Creating test dataset: {test_file}")

        test_data = [
            {
                "input": "RSI: 25, MACD bearish, Price below SMA_20",
                "output": "<analysis>RSI oversold at 25, MACD bearish, price below SMA_20 suggests downtrend</analysis><signal>WEAK_SELL</signal>"
            },
            {
                "input": "RSI: 78, Volume spike, Price at resistance",
                "output": "<analysis>RSI overbought at 78, high volume at resistance suggests reversal</analysis><signal>WEAK_SELL</signal>"
            },
            {
                "input": "MACD bullish crossover, RSI: 45, Price above SMA_50",
                "output": "<analysis>MACD bullish, RSI neutral, price above SMA_50 indicates uptrend</analysis><signal>WEAK_BUY</signal>"
            }
        ]

        # Save test data as CSV
        import csv
        os.makedirs(os.path.dirname(test_file) or '.', exist_ok=True)
        with open(test_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['input', 'output'])
            writer.writeheader()
            writer.writerows(test_data)

        print(f"✓ Created test dataset with {len(test_data)} samples")

    # Load session
    registry = SessionRegistry()
    session_info = registry.get_session(session_id)

    if not session_info or not session_info.checkpoint_path:
        print("❌ Session not found or no checkpoint available")
        return False

    # Create model tester and load model
    tester = ModelTester()
    success, error = tester.load_trained_model(session_info.checkpoint_path, session_id)

    if not success:
        print(f"❌ Failed to load model: {error}")
        return False

    print("✓ Model loaded successfully")

    # Create batch test runner
    batch_runner = BatchTestRunner()

    # Start batch test with reward evaluation
    print(f"\nStarting batch test with reward evaluation...")
    batch_id = batch_runner.start_batch_test(
        model=session_id,
        prompts_file=test_file,
        parameters={
            'temperature': 0.7,
            'max_new_tokens': 512,
            'top_p': 0.95
        },
        model_tester=tester,
        session_info=session_info,
        evaluate_rewards=True
    )

    print(f"✓ Batch test started: {batch_id}")

    # Wait for completion
    import time
    while True:
        status = batch_runner.get_status(batch_id)
        if status and status['status'] in ['completed', 'completed_with_errors', 'failed', 'cancelled']:
            break
        time.sleep(1)

    # Get final results
    status = batch_runner.get_status(batch_id)
    results = batch_runner.get_results(batch_id)

    print(f"\n✓ Batch test completed: {status['status']}")
    print(f"  Successful: {status['successful']}/{status['total']}")
    print(f"  Failed: {status['failed']}")
    print(f"  Average time: {status['average_time']:.2f}s")

    if status.get('average_reward') is not None:
        print(f"\n✓ Reward Statistics:")
        print(f"  Average: {status['average_reward']:.3f}")
        print(f"  Min: {status['min_reward']:.3f}")
        print(f"  Max: {status['max_reward']:.3f}")
        print(f"  Std Dev: {status['reward_std']:.3f}")
    else:
        print("\n⚠️  No reward statistics available")

    if results:
        print(f"\nResults saved to: {status['results_file']}")

    print("\n✓ TEST 3 PASSED\n")
    return True


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description='Test reward integration')
    parser.add_argument('--session-id', required=True, help='Session ID to test with')
    parser.add_argument('--test-file', help='Path to test prompts CSV (optional)')
    parser.add_argument('--skip-batch', action='store_true', help='Skip batch test')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("REWARD INTEGRATION TEST SUITE")
    print("=" * 80 + "\n")

    results = []

    # Test 1: Load reward config
    results.append(("Load Reward Config", test_load_reward_config(args.session_id)))

    # Test 2: Single test with reward
    results.append(("Single Test with Reward", test_single_test_with_reward(args.session_id)))

    # Test 3: Batch test with rewards (optional)
    if not args.skip_batch:
        results.append(("Batch Test with Rewards", test_batch_test_with_rewards(args.session_id, args.test_file)))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
