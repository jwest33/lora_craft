"""Custom reward framework for GRPO training."""

import re
import ast
import math
import json
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import logging

from utils.validators import Validators
from utils.logging_config import get_logger


logger = get_logger(__name__)


class RewardType(Enum):
    """Types of reward functions."""
    BINARY = "binary"  # 0 or 1
    CONTINUOUS = "continuous"  # 0.0 to 1.0
    SPARSE = "sparse"  # Mostly 0 with occasional rewards
    DENSE = "dense"  # Continuous feedback
    CUSTOM = "custom"  # User-defined


@dataclass
class RewardConfig:
    """Configuration for a reward function."""
    name: str
    description: str
    type: RewardType
    weight: float = 1.0

    # Format matching
    use_regex: bool = False
    regex_pattern: Optional[str] = None

    # Numerical comparison
    extract_number: bool = False
    number_tolerance: float = 1e-6
    relative_tolerance: bool = False

    # Length penalties
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    optimal_length: Optional[int] = None

    # Custom function
    custom_code: Optional[str] = None
    safe_mode: bool = True


class RewardFunction:
    """Base class for reward functions."""

    def __init__(self, config: RewardConfig):
        """Initialize reward function.

        Args:
            config: Reward configuration
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate reward configuration."""
        if self.config.use_regex and self.config.regex_pattern:
            try:
                re.compile(self.config.regex_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        if self.config.custom_code:
            valid, msg = Validators.validate_python_code(
                self.config.custom_code,
                safe_mode=self.config.safe_mode
            )
            if not valid:
                raise ValueError(f"Invalid custom code: {msg}")

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None) -> float:
        """Compute reward for generated text.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response (optional)

        Returns:
            Reward value
        """
        raise NotImplementedError


class BinaryReward(RewardFunction):
    """Binary reward (0 or 1) based on exact or pattern matching."""

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None) -> float:
        """Compute binary reward."""
        if reference is None:
            return 0.0

        if self.config.use_regex and self.config.regex_pattern:
            # Use regex matching
            pattern = re.compile(self.config.regex_pattern)
            gen_match = bool(pattern.search(generated))
            ref_match = bool(pattern.search(reference))
            return 1.0 if gen_match == ref_match else 0.0
        else:
            # Exact matching
            gen_normalized = generated.strip().lower()
            ref_normalized = reference.strip().lower()
            return 1.0 if gen_normalized == ref_normalized else 0.0


class NumericalReward(RewardFunction):
    """Reward based on numerical answer extraction and comparison."""

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text."""
        # Pattern to match numbers (including decimals and scientific notation)
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                num = float(match)
                numbers.append(num)
            except ValueError:
                continue

        return numbers

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None) -> float:
        """Compute numerical reward."""
        if reference is None:
            return 0.0

        gen_numbers = self._extract_numbers(generated)
        ref_numbers = self._extract_numbers(reference)

        if not ref_numbers:
            # No numbers in reference
            return 1.0 if not gen_numbers else 0.0

        if not gen_numbers:
            # No numbers extracted from generation
            return 0.0

        # Compare first number (or all numbers)
        ref_num = ref_numbers[0]
        gen_num = gen_numbers[0]

        if self.config.relative_tolerance:
            # Relative tolerance
            if ref_num == 0:
                is_close = abs(gen_num) < self.config.number_tolerance
            else:
                rel_error = abs(gen_num - ref_num) / abs(ref_num)
                is_close = rel_error < self.config.number_tolerance
        else:
            # Absolute tolerance
            is_close = abs(gen_num - ref_num) < self.config.number_tolerance

        return 1.0 if is_close else 0.0


class LengthReward(RewardFunction):
    """Reward based on response length."""

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None) -> float:
        """Compute length-based reward."""
        gen_length = len(generated.split())

        reward = 1.0

        # Check minimum length
        if self.config.min_length and gen_length < self.config.min_length:
            penalty = (self.config.min_length - gen_length) / self.config.min_length
            reward *= (1 - penalty)

        # Check maximum length
        if self.config.max_length and gen_length > self.config.max_length:
            penalty = (gen_length - self.config.max_length) / self.config.max_length
            reward *= (1 - min(penalty, 0.5))  # Cap penalty at 50%

        # Check optimal length
        if self.config.optimal_length:
            distance = abs(gen_length - self.config.optimal_length)
            penalty = distance / self.config.optimal_length
            reward *= math.exp(-penalty)  # Gaussian-like penalty

        return max(0.0, min(1.0, reward))


class FormatReward(RewardFunction):
    """Reward based on format compliance."""

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None) -> float:
        """Compute format-based reward."""
        if not self.config.regex_pattern:
            return 1.0

        pattern = re.compile(self.config.regex_pattern, re.MULTILINE | re.DOTALL)
        match = pattern.search(generated)

        return 1.0 if match else 0.0


class CustomReward(RewardFunction):
    """Custom user-defined reward function."""

    def __init__(self, config: RewardConfig):
        """Initialize custom reward."""
        super().__init__(config)
        self._compile_function()

    def _compile_function(self):
        """Compile custom reward function."""
        if not self.config.custom_code:
            raise ValueError("Custom code is required for custom reward")

        # Create a restricted namespace
        namespace = {
            're': re,
            'math': math,
            'np': np,
            'len': len,
            'str': str,
            'float': float,
            'int': int,
            'abs': abs,
            'min': min,
            'max': max,
        }

        # Compile and execute the code
        try:
            exec(self.config.custom_code, namespace)

            # Look for compute_reward function
            if 'compute_reward' not in namespace:
                raise ValueError("Custom code must define compute_reward(instruction, generated, reference) function")

            self._custom_func = namespace['compute_reward']

        except Exception as e:
            raise ValueError(f"Failed to compile custom code: {e}")

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None) -> float:
        """Compute custom reward."""
        try:
            reward = self._custom_func(instruction, generated, reference)
            return float(max(0.0, min(1.0, reward)))
        except Exception as e:
            logger.error(f"Custom reward computation failed: {e}")
            return 0.0


class CustomRewardBuilder:
    """Builder for creating and combining custom reward functions."""

    def __init__(self):
        """Initialize reward builder."""
        self.rewards: List[RewardFunction] = []
        self.weights: List[float] = []

    def add_reward(self, reward: RewardFunction, weight: float = 1.0):
        """Add a reward function.

        Args:
            reward: Reward function
            weight: Weight for this reward
        """
        self.rewards.append(reward)
        self.weights.append(weight)

    def add_binary_reward(self, name: str, regex_pattern: Optional[str] = None, weight: float = 1.0):
        """Add binary reward function."""
        config = RewardConfig(
            name=name,
            description=f"Binary reward: {name}",
            type=RewardType.BINARY,
            use_regex=bool(regex_pattern),
            regex_pattern=regex_pattern,
            weight=weight
        )
        self.add_reward(BinaryReward(config), weight)

    def add_numerical_reward(self, name: str, tolerance: float = 1e-6, relative: bool = False, weight: float = 1.0):
        """Add numerical reward function."""
        config = RewardConfig(
            name=name,
            description=f"Numerical reward: {name}",
            type=RewardType.CONTINUOUS,
            extract_number=True,
            number_tolerance=tolerance,
            relative_tolerance=relative,
            weight=weight
        )
        self.add_reward(NumericalReward(config), weight)

    def add_length_reward(self, name: str,
                         min_length: Optional[int] = None,
                         max_length: Optional[int] = None,
                         optimal_length: Optional[int] = None,
                         weight: float = 1.0):
        """Add length-based reward function."""
        config = RewardConfig(
            name=name,
            description=f"Length reward: {name}",
            type=RewardType.CONTINUOUS,
            min_length=min_length,
            max_length=max_length,
            optimal_length=optimal_length,
            weight=weight
        )
        self.add_reward(LengthReward(config), weight)

    def add_format_reward(self, name: str, pattern: str, weight: float = 1.0):
        """Add format compliance reward."""
        config = RewardConfig(
            name=name,
            description=f"Format reward: {name}",
            type=RewardType.BINARY,
            use_regex=True,
            regex_pattern=pattern,
            weight=weight
        )
        self.add_reward(FormatReward(config), weight)

    def add_custom_reward(self, name: str, code: str, weight: float = 1.0, safe_mode: bool = True):
        """Add custom reward function."""
        config = RewardConfig(
            name=name,
            description=f"Custom reward: {name}",
            type=RewardType.CUSTOM,
            custom_code=code,
            safe_mode=safe_mode,
            weight=weight
        )
        self.add_reward(CustomReward(config), weight)

    def compute_total_reward(self,
                            instruction: str,
                            generated: str,
                            reference: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        """Compute total reward from all components.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response

        Returns:
            Tuple of (total_reward, component_rewards)
        """
        if not self.rewards:
            return 0.0, {}

        component_rewards = {}
        weighted_sum = 0.0
        total_weight = sum(self.weights)

        for reward_func, weight in zip(self.rewards, self.weights):
            reward_value = reward_func.compute(instruction, generated, reference)
            component_rewards[reward_func.config.name] = reward_value
            weighted_sum += reward_value * weight

        total_reward = weighted_sum / total_weight if total_weight > 0 else 0.0

        return total_reward, component_rewards

    def test_reward(self,
                   test_cases: List[Tuple[str, str, str, float]]) -> Dict[str, Any]:
        """Test reward function with examples.

        Args:
            test_cases: List of (instruction, generated, reference, expected_reward)

        Returns:
            Test results
        """
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for instruction, generated, reference, expected in test_cases:
            total, components = self.compute_total_reward(instruction, generated, reference)

            # Check if reward is close to expected
            is_close = abs(total - expected) < 0.01

            result = {
                'instruction': instruction[:50] + '...' if len(instruction) > 50 else instruction,
                'generated': generated[:50] + '...' if len(generated) > 50 else generated,
                'expected': expected,
                'actual': total,
                'components': components,
                'passed': is_close
            }

            results['details'].append(result)

            if is_close:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def save_config(self, path: str):
        """Save reward configuration to file."""
        config = {
            'rewards': [
                {
                    'config': asdict(r.config),
                    'weight': w,
                    'class': r.__class__.__name__
                }
                for r, w in zip(self.rewards, self.weights)
            ]
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> 'CustomRewardBuilder':
        """Load reward configuration from file."""
        with open(path, 'r') as f:
            config = json.load(f)

        builder = cls()

        for reward_info in config['rewards']:
            reward_config = RewardConfig(**reward_info['config'])
            weight = reward_info['weight']
            class_name = reward_info['class']

            # Create appropriate reward instance
            if class_name == 'BinaryReward':
                reward = BinaryReward(reward_config)
            elif class_name == 'NumericalReward':
                reward = NumericalReward(reward_config)
            elif class_name == 'LengthReward':
                reward = LengthReward(reward_config)
            elif class_name == 'FormatReward':
                reward = FormatReward(reward_config)
            elif class_name == 'CustomReward':
                reward = CustomReward(reward_config)
            else:
                logger.warning(f"Unknown reward class: {class_name}")
                continue

            builder.add_reward(reward, weight)

        return builder


# Preset reward functions
def create_math_reward() -> CustomRewardBuilder:
    """Create reward for mathematical problems."""
    builder = CustomRewardBuilder()

    # Numerical accuracy
    builder.add_numerical_reward("numerical_accuracy", tolerance=1e-6, weight=0.7)

    # Format compliance (answer in a box)
    builder.add_format_reward(
        "answer_format",
        pattern=r"\\boxed\{[^}]+\}|Final Answer:.*|Answer:.*",
        weight=0.2
    )

    # Length penalty (prefer concise solutions)
    builder.add_length_reward(
        "length",
        min_length=10,
        max_length=500,
        optimal_length=100,
        weight=0.1
    )

    return builder


def create_code_reward() -> CustomRewardBuilder:
    """Create reward for code generation."""
    builder = CustomRewardBuilder()

    # Code block detection
    builder.add_format_reward(
        "code_block",
        pattern=r"```[\w]*\n.*?\n```",
        weight=0.3
    )

    # Function definition detection
    builder.add_format_reward(
        "function_def",
        pattern=r"def\s+\w+\s*\(|function\s+\w+\s*\(|const\s+\w+\s*=",
        weight=0.3
    )

    # Comments detection
    builder.add_format_reward(
        "comments",
        pattern=r"#.*|//.*|/\*.*?\*/",
        weight=0.2
    )

    # Length reward (prefer reasonable length)
    builder.add_length_reward(
        "length",
        min_length=20,
        max_length=1000,
        optimal_length=200,
        weight=0.2
    )

    return builder


if __name__ == "__main__":
    # Test reward functions
    builder = create_math_reward()

    test_cases = [
        ("What is 2+2?", "The answer is 4", "4", 0.7),
        ("Solve x^2 = 16", "x = 4 or x = -4\n\\boxed{4, -4}", "x = 4, x = -4", 0.9),
        ("Calculate 10/3", "3.333333", "3.333333", 0.7),
    ]

    results = builder.test_reward(test_cases)
    print(f"Test results: {results['passed']}/{len(test_cases)} passed")

    for detail in results['details']:
        print(f"\nInstruction: {detail['instruction']}")
        print(f"Expected: {detail['expected']:.2f}, Actual: {detail['actual']:.2f}")
        print(f"Components: {detail['components']}")
