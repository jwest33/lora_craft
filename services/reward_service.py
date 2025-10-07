"""
Reward service for reward function management and testing.

This module provides business logic for:
- Reward preset management
- Reward template management
- Reward testing and validation
- Custom reward preset CRUD operations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from core import CustomRewardBuilder
from core.custom_rewards import (
    RewardPresetLibrary,
    RewardTester,
    RewardTemplateLibrary
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RewardService:
    """Service for managing reward functions and presets."""

    def __init__(self, presets_folder: str = 'presets'):
        """
        Initialize reward service.

        Args:
            presets_folder: Path to folder containing custom presets
        """
        self.presets_folder = presets_folder
        self.custom_presets_file = os.path.join(presets_folder, 'custom_presets.json')

        # Ensure presets directory exists
        Path(presets_folder).mkdir(exist_ok=True)

    def get_all_presets(self) -> Dict[str, Any]:
        """
        Get all available reward presets with metadata.

        Returns:
            Dictionary of presets with their metadata
        """
        try:
            library = RewardPresetLibrary()
            return library.to_dict()
        except Exception as e:
            logger.error(f"Failed to load reward presets: {e}")
            raise

    def get_preset_details(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed component breakdown for a preset reward.

        Args:
            preset_name: Name of the preset to get details for

        Returns:
            Dictionary with preset details and components, or None if not found
        """
        try:
            library = RewardPresetLibrary()
            preset = library.get_preset(preset_name)

            if not preset:
                return None

            # Create the builder to get component details
            builder = preset.create_builder()
            components = builder.get_component_details()

            # Calculate total weight for validation
            total_weight = sum(c['weight'] for c in components)

            # Get preset metadata
            preset_data = preset.to_dict()

            return {
                'name': preset_data['name'],
                'description': preset_data['description'],
                'example_input': preset_data['example_input'],
                'example_output': preset_data['example_output'],
                'difficulty': preset_data['difficulty'],
                'tags': preset_data['tags'],
                'components': components,
                'total_weight': total_weight,
                'weight_valid': abs(total_weight - 1.0) < 0.001
            }
        except Exception as e:
            logger.error(f"Failed to get preset details for {preset_name}: {e}")
            raise

    def get_all_templates(self) -> Dict[str, Any]:
        """
        Get all reward templates for quick start.

        Returns:
            Dictionary of templates
        """
        try:
            library = RewardTemplateLibrary()
            templates = {
                name: template.to_dict()
                for name, template in library.get_all_templates().items()
            }
            return {'templates': templates}
        except Exception as e:
            logger.error(f"Failed to load reward templates: {e}")
            raise

    def build_reward_from_config(self, reward_config: Dict[str, Any]) -> CustomRewardBuilder:
        """
        Build a reward function from configuration.

        Args:
            reward_config: Reward configuration dictionary

        Returns:
            CustomRewardBuilder instance

        Raises:
            ValueError: If configuration is invalid
        """
        reward_builder = CustomRewardBuilder()

        if reward_config.get('type') == 'preset':
            # Use preset
            library = RewardPresetLibrary()
            preset_name = reward_config.get('preset_name')
            preset = library.get_preset(preset_name)

            if not preset:
                raise ValueError(f'Unknown preset: {preset_name}')

            reward_builder = preset.create_builder()
        else:
            # Build custom reward
            components = reward_config.get('components', [])

            for idx, comp in enumerate(components):
                comp_type = comp.get('type')
                comp_name = comp.get('name', f"{comp_type}_{idx}")
                weight = comp.get('weight', 1.0)
                parameters = comp.get('parameters', {})

                if comp_type == 'binary':
                    pattern = parameters.get('pattern') or comp.get('pattern')
                    reward_builder.add_binary_reward(
                        comp_name,
                        regex_pattern=pattern,
                        weight=weight
                    )
                elif comp_type == 'numerical':
                    tolerance = parameters.get('tolerance', 1e-6)
                    reward_builder.add_numerical_reward(
                        comp_name,
                        tolerance=tolerance,
                        weight=weight
                    )
                elif comp_type == 'length':
                    min_len = parameters.get('min_length')
                    max_len = parameters.get('max_length')
                    optimal_len = parameters.get('optimal_length')
                    reward_builder.add_length_reward(
                        comp_name,
                        min_length=min_len,
                        max_length=max_len,
                        optimal_length=optimal_len,
                        weight=weight
                    )
                elif comp_type == 'format':
                    pattern = parameters.get('pattern') or comp.get('pattern', r".*")
                    reward_builder.add_format_reward(
                        comp_name,
                        pattern=pattern,
                        weight=weight
                    )

        return reward_builder

    def test_reward(self, reward_config: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test a reward configuration with sample inputs.

        Args:
            reward_config: Reward configuration to test
            test_cases: List of test cases with instruction, generated, and optional reference

        Returns:
            Dictionary with test results and statistics

        Raises:
            ValueError: If configuration or test cases are invalid
        """
        if not reward_config:
            raise ValueError('No reward configuration provided')

        if not test_cases:
            raise ValueError('No test cases provided')

        # Build the reward from configuration
        reward_builder = self.build_reward_from_config(reward_config)

        # Test the reward
        tester = RewardTester(reward_builder)

        # Run batch test if multiple cases
        if len(test_cases) > 1:
            results = tester.test_batch(test_cases)
        else:
            # Single test
            case = test_cases[0]
            result = tester.test_single(
                case.get('instruction', ''),
                case.get('generated', ''),
                case.get('reference')
            )
            results = {
                'results': [result],
                'statistics': {
                    'mean': result['total_reward'],
                    'std': 0,
                    'min': result['total_reward'],
                    'max': result['total_reward']
                }
            }

        return results

    def compare_rewards(self, reward_configs: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple reward configurations on the same test cases.

        Args:
            reward_configs: List of reward configurations to compare
            test_cases: List of test cases to evaluate

        Returns:
            Dictionary with comparison results
        """
        if not reward_configs or len(reward_configs) < 2:
            raise ValueError('Need at least 2 reward configurations to compare')

        if not test_cases:
            raise ValueError('No test cases provided')

        results = []
        for config in reward_configs:
            try:
                reward_builder = self.build_reward_from_config(config)
                tester = RewardTester(reward_builder)
                test_result = tester.test_batch(test_cases)

                results.append({
                    'config': config,
                    'results': test_result
                })
            except Exception as e:
                logger.error(f"Error testing reward config: {e}")
                results.append({
                    'config': config,
                    'error': str(e)
                })

        return {'comparisons': results}

    def validate_reward_fields(self, reward_config: Dict[str, Any], sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that a reward configuration can work with given sample data.

        Args:
            reward_config: Reward configuration to validate
            sample_data: Sample data with instruction and response fields

        Returns:
            Dictionary with validation results
        """
        try:
            # Try to build the reward
            reward_builder = self.build_reward_from_config(reward_config)

            # Try to evaluate it
            tester = RewardTester(reward_builder)
            result = tester.test_single(
                sample_data.get('instruction', ''),
                sample_data.get('response', ''),
                sample_data.get('reference')
            )

            return {
                'valid': True,
                'sample_result': result
            }
        except Exception as e:
            logger.error(f"Reward validation failed: {e}")
            return {
                'valid': False,
                'error': str(e)
            }

    def get_custom_presets(self) -> Dict[str, Any]:
        """
        Get all custom (user-created) presets.

        Returns:
            Dictionary of custom presets
        """
        try:
            if not os.path.exists(self.custom_presets_file):
                return {'presets': {}}

            with open(self.custom_presets_file, 'r') as f:
                data = json.load(f)
                return {'presets': data}
        except Exception as e:
            logger.error(f"Failed to load custom presets: {e}")
            raise

    def save_custom_preset(self, preset_name: str, preset_config: Dict[str, Any]) -> bool:
        """
        Save a custom reward preset.

        Args:
            preset_name: Name for the custom preset
            preset_config: Preset configuration

        Returns:
            True if saved successfully

        Raises:
            ValueError: If preset name or config is invalid
        """
        if not preset_name or not preset_name.strip():
            raise ValueError('Preset name cannot be empty')

        if not preset_config:
            raise ValueError('Preset configuration cannot be empty')

        try:
            # Load existing presets
            presets = {}
            if os.path.exists(self.custom_presets_file):
                with open(self.custom_presets_file, 'r') as f:
                    presets = json.load(f)

            # Add/update preset
            presets[preset_name] = preset_config

            # Save back to file
            with open(self.custom_presets_file, 'w') as f:
                json.dump(presets, f, indent=2)

            logger.info(f"Saved custom preset: {preset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save custom preset {preset_name}: {e}")
            raise

    def delete_custom_preset(self, preset_name: str) -> bool:
        """
        Delete a custom reward preset.

        Args:
            preset_name: Name of the preset to delete

        Returns:
            True if deleted successfully, False if not found

        Raises:
            Exception: If deletion fails
        """
        try:
            if not os.path.exists(self.custom_presets_file):
                return False

            with open(self.custom_presets_file, 'r') as f:
                presets = json.load(f)

            if preset_name not in presets:
                return False

            del presets[preset_name]

            with open(self.custom_presets_file, 'w') as f:
                json.dump(presets, f, indent=2)

            logger.info(f"Deleted custom preset: {preset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete custom preset {preset_name}: {e}")
            raise

    def update_custom_preset(self, preset_name: str, preset_config: Dict[str, Any]) -> bool:
        """
        Update an existing custom reward preset.

        Args:
            preset_name: Name of the preset to update
            preset_config: New preset configuration

        Returns:
            True if updated successfully, False if preset doesn't exist

        Raises:
            Exception: If update fails
        """
        try:
            if not os.path.exists(self.custom_presets_file):
                return False

            with open(self.custom_presets_file, 'r') as f:
                presets = json.load(f)

            if preset_name not in presets:
                return False

            presets[preset_name] = preset_config

            with open(self.custom_presets_file, 'w') as f:
                json.dump(presets, f, indent=2)

            logger.info(f"Updated custom preset: {preset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update custom preset {preset_name}: {e}")
            raise
