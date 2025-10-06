"""Utility module for loading and creating reward functions from session configs."""

import logging
from typing import Dict, Any, Optional, List
from core.custom_rewards import (
    CustomRewardBuilder,
    RewardConfig,
    RewardType,
    SignalAccuracyReward,
    TemplateValidationReward,
    LengthReward,
    FormatReward,
    BinaryReward,
    NumericalReward,
    MultiChoiceValidationReward,
    SectionContentReward,
    SequentialPatternReward,
    CustomReward
)

logger = logging.getLogger(__name__)


def load_reward_from_session(session_info) -> Optional[CustomRewardBuilder]:
    """Load reward function from a SessionInfo object.

    Args:
        session_info: SessionInfo object containing training_config with reward_config

    Returns:
        CustomRewardBuilder instance or None if no reward config found
    """
    if not session_info:
        logger.warning("No session_info provided")
        return None

    if not hasattr(session_info, 'training_config') or not session_info.training_config:
        logger.warning("No training_config in session_info")
        return None

    reward_config = session_info.training_config.get('reward_config')
    if not reward_config:
        logger.warning("No reward_config in training_config")
        return None

    return load_reward_from_config(reward_config)


def load_reward_from_config(reward_config: Dict[str, Any]) -> Optional[CustomRewardBuilder]:
    """Load reward function from a reward configuration dictionary.

    Args:
        reward_config: Dictionary containing reward configuration with 'components' list

    Returns:
        CustomRewardBuilder instance or None if invalid config

    Example reward_config structure:
        {
            "type": "custom",
            "components": [
                {
                    "name": "signal_accuracy",
                    "type": "continuous",
                    "weight": 0.85,
                    "parameters": {
                        "direction_match_score": 0.7,
                        "valid_signals": ["STRONG_BUY", "WEAK_BUY", "HOLD", ...]
                    }
                },
                ...
            ]
        }
    """
    try:
        if not reward_config or not isinstance(reward_config, dict):
            logger.warning("Invalid reward_config: not a dictionary")
            return None

        components = reward_config.get('components', [])
        if not components:
            logger.warning("No components in reward_config")
            return None

        builder = CustomRewardBuilder()

        for component in components:
            try:
                _add_component_to_builder(builder, component)
            except Exception as e:
                logger.error(f"Failed to add reward component '{component.get('name', 'unknown')}': {e}")
                # Continue with other components

        if not builder.rewards:
            logger.warning("No reward components were successfully loaded")
            return None

        logger.info(f"Successfully loaded {len(builder.rewards)} reward components")
        return builder

    except Exception as e:
        logger.error(f"Failed to load reward config: {e}")
        return None


def _add_component_to_builder(builder: CustomRewardBuilder, component: Dict[str, Any]) -> None:
    """Add a single reward component to the builder.

    Args:
        builder: CustomRewardBuilder instance to add to
        component: Component configuration dictionary
    """
    name = component.get('name', 'unnamed')
    comp_type = component.get('type', 'continuous')
    weight = component.get('weight', 1.0)
    parameters = component.get('parameters', {})

    # Map component names to builder methods
    if name == 'signal_accuracy':
        _add_signal_accuracy(builder, name, weight, parameters)
    elif name == 'template_structure' or name == 'template_validation':
        _add_template_validation(builder, name, weight, parameters)
    elif name == 'response_length' or name.endswith('_length'):
        _add_length_reward(builder, name, weight, parameters)
    elif name == 'format' or 'format' in name:
        _add_format_reward(builder, name, weight, parameters)
    elif name == 'numerical' or 'numerical' in name:
        _add_numerical_reward(builder, name, weight, parameters)
    elif name == 'multi_choice' or name == 'multichoice_validation':
        _add_multi_choice(builder, name, weight, parameters)
    elif name == 'section_content':
        _add_section_content(builder, name, weight, parameters)
    elif name == 'sequential_pattern':
        _add_sequential_pattern(builder, name, weight, parameters)
    elif name == 'binary':
        _add_binary_reward(builder, name, weight, parameters)
    elif name == 'custom':
        _add_custom_reward(builder, name, weight, parameters)
    else:
        # Try to infer from type
        logger.warning(f"Unknown component name '{name}', attempting to infer from type '{comp_type}'")
        _add_by_type(builder, name, comp_type, weight, parameters)


def _add_signal_accuracy(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add signal accuracy reward component."""
    valid_signals = params.get('valid_signals', ["STRONG_BUY", "WEAK_BUY", "HOLD", "WEAK_SELL", "STRONG_SELL"])
    direction_match_score = params.get('direction_match_score', 0.70)

    builder.add_signal_accuracy(
        name=name,
        valid_signals=valid_signals,
        direction_match_score=direction_match_score,
        weight=weight
    )
    logger.debug(f"Added signal_accuracy component: {name}")


def _add_template_validation(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add template validation reward component."""
    section_tags = params.get('section_tags', [])
    required_sections = params.get('required_sections', section_tags)
    order_matters = params.get('order_matters', False)

    builder.add_template_validation(
        name=name,
        section_tags=section_tags,
        required_sections=required_sections,
        order_matters=order_matters,
        weight=weight
    )
    logger.debug(f"Added template_validation component: {name}")


def _add_length_reward(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add length-based reward component."""
    min_length = params.get('min_length')
    max_length = params.get('max_length')
    optimal_length = params.get('optimal_length')

    builder.add_length_reward(
        name=name,
        min_length=min_length,
        max_length=max_length,
        optimal_length=optimal_length,
        weight=weight
    )
    logger.debug(f"Added length_reward component: {name}")


def _add_format_reward(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add format compliance reward component."""
    pattern = params.get('pattern', params.get('regex_pattern', ''))

    if not pattern:
        logger.warning(f"Format reward '{name}' missing pattern, skipping")
        return

    builder.add_format_reward(
        name=name,
        pattern=pattern,
        weight=weight
    )
    logger.debug(f"Added format_reward component: {name}")


def _add_numerical_reward(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add numerical comparison reward component."""
    tolerance = params.get('tolerance', params.get('number_tolerance', 1e-6))
    relative = params.get('relative', params.get('relative_tolerance', False))

    builder.add_numerical_reward(
        name=name,
        tolerance=tolerance,
        relative=relative,
        weight=weight
    )
    logger.debug(f"Added numerical_reward component: {name}")


def _add_multi_choice(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add multi-choice validation reward component."""
    valid_choices = params.get('valid_choices', [])
    case_sensitive = params.get('case_sensitive', False)
    exact_match = params.get('exact_match', True)

    builder.add_multi_choice_validation(
        name=name,
        valid_choices=valid_choices,
        case_sensitive=case_sensitive,
        exact_match=exact_match,
        weight=weight
    )
    logger.debug(f"Added multi_choice component: {name}")


def _add_section_content(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add section content validation reward component."""
    section_tag = params.get('section_tag', '')
    min_words = params.get('min_words')
    max_words = params.get('max_words')
    required_keywords = params.get('required_keywords', [])

    builder.add_section_content(
        name=name,
        section_tag=section_tag,
        min_words=min_words,
        max_words=max_words,
        required_keywords=required_keywords,
        weight=weight
    )
    logger.debug(f"Added section_content component: {name}")


def _add_sequential_pattern(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add sequential pattern reward component."""
    patterns = params.get('patterns', [])
    strict_order = params.get('strict_order', True)

    builder.add_sequential_pattern(
        name=name,
        patterns=patterns,
        strict_order=strict_order,
        weight=weight
    )
    logger.debug(f"Added sequential_pattern component: {name}")


def _add_binary_reward(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add binary reward component."""
    regex_pattern = params.get('regex_pattern', params.get('pattern'))

    builder.add_binary_reward(
        name=name,
        regex_pattern=regex_pattern,
        weight=weight
    )
    logger.debug(f"Added binary_reward component: {name}")


def _add_custom_reward(builder: CustomRewardBuilder, name: str, weight: float, params: Dict[str, Any]) -> None:
    """Add custom code reward component."""
    code = params.get('custom_code', params.get('code', ''))
    safe_mode = params.get('safe_mode', True)

    if not code:
        logger.warning(f"Custom reward '{name}' missing code, skipping")
        return

    builder.add_custom_reward(
        name=name,
        code=code,
        weight=weight,
        safe_mode=safe_mode
    )
    logger.debug(f"Added custom_reward component: {name}")


def _add_by_type(builder: CustomRewardBuilder, name: str, comp_type: str, weight: float, params: Dict[str, Any]) -> None:
    """Fallback: Add component by inferring from type."""
    if comp_type == 'binary':
        _add_binary_reward(builder, name, weight, params)
    elif comp_type == 'continuous':
        # Default to length reward for continuous type
        _add_length_reward(builder, name, weight, params)
    else:
        logger.warning(f"Cannot infer component type for '{name}' with type '{comp_type}', skipping")


def get_reward_component_details(session_info) -> Optional[List[Dict[str, Any]]]:
    """Get detailed information about reward components from a session.

    Args:
        session_info: SessionInfo object

    Returns:
        List of component detail dictionaries or None
    """
    builder = load_reward_from_session(session_info)
    if not builder:
        return None

    return builder.get_component_details()
