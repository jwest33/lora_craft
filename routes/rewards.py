"""
Reward routes for reward function management and testing.

This module provides endpoints for:
- Reward preset management (built-in and custom)
- Reward template quick-start
- Reward testing and validation
- Reward comparison and visualization
"""

from difflib import SequenceMatcher
from flask import Blueprint, request, jsonify, current_app

from services import RewardService
from core import ModelTester, CustomRewardBuilder
from core.custom_rewards import RewardPresetLibrary, RewardTester
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create blueprint
rewards_bp = Blueprint('rewards', __name__, url_prefix='/api/rewards')

# Initialize service
reward_service = RewardService()


@rewards_bp.route('/presets', methods=['GET'])
def get_reward_presets():
    """Get all available reward presets with metadata."""
    try:
        return jsonify(reward_service.get_all_presets())
    except Exception as e:
        logger.error(f"Failed to load reward presets: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/preset-details/<preset_name>', methods=['GET'])
def get_preset_details(preset_name):
    """Get detailed component breakdown for a preset reward."""
    try:
        details = reward_service.get_preset_details(preset_name)

        if not details:
            return jsonify({'error': f'Preset not found: {preset_name}'}), 404

        return jsonify(details)
    except Exception as e:
        logger.error(f"Failed to get preset details for {preset_name}: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/templates', methods=['GET'])
def get_reward_templates():
    """Get all reward templates for quick start."""
    try:
        return jsonify(reward_service.get_all_templates())
    except Exception as e:
        logger.error(f"Failed to load reward templates: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/test', methods=['POST'])
def test_reward():
    """Test a reward configuration with sample inputs."""
    try:
        data = request.get_json()
        reward_config = data.get('reward_config')
        test_cases = data.get('test_cases', [])

        results = reward_service.test_reward(reward_config, test_cases)
        return jsonify(results)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to test reward: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/test-with-model', methods=['POST'])
def test_reward_with_model():
    """Test reward function by generating a response with a model and scoring it."""
    try:
        data = request.get_json()
        reward_config = data.get('reward_config')
        instruction = data.get('instruction')
        reference = data.get('reference')
        model_type = data.get('model_type', 'trained')  # 'trained' or 'base'
        model_key = data.get('model_key')  # session_id for trained, model_name for base
        generation_config = data.get('generation_config', {})
        system_prompt = data.get('system_prompt', '')

        if not reward_config or not instruction or not model_key:
            return jsonify({'error': 'Missing required parameters (reward_config, instruction, model_key)'}), 400

        # Build the reward from configuration
        reward_builder = reward_service.build_reward_from_config(reward_config)

        # Get session info if testing trained model
        session_registry = current_app.session_registry
        session_info = None
        if model_type == 'trained':
            session_info = session_registry.get_session(model_key)

        # Create tester and test with model
        tester = RewardTester(reward_builder)

        # Create model tester instance
        model_tester = ModelTester()

        # Load the appropriate model
        if model_type == 'trained':
            # Load trained model from checkpoint
            checkpoint_path = session_registry.get_checkpoint_path(model_key)
            if not checkpoint_path:
                return jsonify({'error': f'No checkpoint found for session {model_key}'}), 404

            success, error = model_tester.load_trained_model(checkpoint_path, model_key)
            if not success:
                return jsonify({'error': f'Failed to load model: {error}'}), 500
        else:
            # Load base model
            success, error = model_tester.load_base_model(model_key)
            if not success:
                return jsonify({'error': f'Failed to load model: {error}'}), 500

        # Test with the loaded model
        result = tester.test_with_model(
            instruction=instruction,
            reference=reference,
            model_tester=model_tester,
            model_type=model_type,
            model_key=model_key,
            generation_config=generation_config,
            session_info=session_info,
            system_prompt=system_prompt
        )

        if not result.get('success', False):
            return jsonify({'error': result.get('error', 'Unknown error')}), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to test reward with model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/compare', methods=['POST'])
def compare_rewards():
    """Compare multiple responses using a reward function."""
    try:
        data = request.get_json()
        reward_config = data.get('reward_config')
        instruction = data.get('instruction')
        responses = data.get('responses', [])
        reference = data.get('reference')

        if not reward_config or not instruction or not responses:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Build the reward
        reward_builder = reward_service.build_reward_from_config(reward_config)

        # Compare responses
        tester = RewardTester(reward_builder)
        results = tester.compare_responses(instruction, responses, reference)

        return jsonify({'comparisons': results})

    except Exception as e:
        logger.error(f"Failed to compare responses: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/visualize', methods=['POST'])
def visualize_reward():
    """Get visualization data for a reward test result."""
    try:
        data = request.get_json()
        test_result = data.get('test_result')

        if not test_result:
            return jsonify({'error': 'No test result provided'}), 400

        # Create visualization
        tester = RewardTester(CustomRewardBuilder())  # Dummy builder for viz
        visualization = tester.visualize_components(test_result)

        return jsonify({
            'visualization': visualization,
            'chart_data': {
                'labels': list(test_result.get('components', {}).keys()),
                'values': list(test_result.get('components', {}).values())
            }
        })

    except Exception as e:
        logger.error(f"Failed to visualize reward: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/recommend', methods=['POST'])
def recommend_reward():
    """Recommend a reward configuration based on task description."""
    try:
        data = request.get_json()
        task_description = data.get('task_description', '')
        example_input = data.get('example_input', '')
        example_output = data.get('example_output', '')

        # Simple keyword-based recommendation
        library = RewardPresetLibrary()

        # Search for matching presets
        keywords = task_description.lower().split()
        recommendations = []

        for preset_name, preset in library.presets.items():
            score = 0

            # Check for keyword matches
            for keyword in keywords:
                if keyword in preset.name.lower():
                    score += 2
                if keyword in preset.description.lower():
                    score += 1
                if any(keyword in tag.lower() for tag in preset.tags):
                    score += 1

            if score > 0:
                recommendations.append({
                    'preset': preset.to_dict(),
                    'score': score,
                    'reason': f"Matches {score} keywords from your description"
                })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        # Also recommend templates
        from core.custom_rewards import RewardTemplateLibrary
        template_library = RewardTemplateLibrary()
        template_recommendations = []

        for template_name, template in template_library.templates.items():
            if any(keyword in template.description.lower() for keyword in keywords):
                template_recommendations.append(template.to_dict())

        return jsonify({
            'preset_recommendations': recommendations[:5],  # Top 5
            'template_recommendations': template_recommendations[:3]  # Top 3
        })

    except Exception as e:
        logger.error(f"Failed to recommend reward: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/validate-fields', methods=['POST'])
def validate_reward_fields():
    """Validate and suggest field mappings between dataset and reward function."""
    try:
        data = request.get_json()
        preset_name = data.get('reward_preset')
        dataset_columns = data.get('dataset_columns', [])
        current_mapping = data.get('current_mapping', {})

        if not preset_name or not dataset_columns:
            return jsonify({'error': 'Missing reward_preset or dataset_columns'}), 400

        # Get reward preset metadata
        library = RewardPresetLibrary()
        preset = library.get_preset(preset_name)

        if not preset:
            return jsonify({'error': f'Preset not found: {preset_name}'}), 404

        # Helper function for fuzzy string matching
        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        # Generate field mapping suggestions
        suggestions = {}
        confidence_scores = {}
        warnings = []

        # Expected fields from the reward preset
        expected_fields = preset.expected_fields
        optional_fields = preset.optional_fields or []

        for field_name, field_description in expected_fields.items():
            best_match = None
            best_score = 0.0

            # Try exact match first
            if field_name in dataset_columns:
                best_match = field_name
                best_score = 1.0
            else:
                # Try fuzzy matching with common patterns
                field_patterns = {
                    'instruction': ['instruction', 'prompt', 'question', 'input', 'problem', 'query', 'user'],
                    'response': ['response', 'answer', 'output', 'completion', 'solution', 'reply', 'assistant'],
                    'reference': ['reference', 'expected', 'target', 'ground_truth', 'label']
                }

                patterns = field_patterns.get(field_name, [field_name])

                for col in dataset_columns:
                    col_lower = col.lower()

                    # Check pattern matches
                    for pattern in patterns:
                        if pattern in col_lower or col_lower in pattern:
                            score = similarity(pattern, col_lower)
                            if score > best_score:
                                best_score = score
                                best_match = col

                    # Also check direct similarity
                    score = similarity(field_name, col)
                    if score > best_score:
                        best_score = score
                        best_match = col

            if best_match:
                suggestions[field_name] = best_match
                confidence_scores[field_name] = best_score

                # Add warning for low confidence matches
                if best_score < 0.7 and field_name not in optional_fields:
                    warnings.append({
                        'type': 'low_confidence',
                        'field': field_name,
                        'message': f"Low confidence match for '{field_name}' → '{best_match}' ({int(best_score * 100)}%)"
                    })
            elif field_name not in optional_fields:
                # Required field not found
                warnings.append({
                    'type': 'missing_required',
                    'field': field_name,
                    'message': f"Required field '{field_name}' not found in dataset. Expected: {field_description}"
                })

        # Check for unmapped dataset columns
        mapped_columns = set(suggestions.values())
        unmapped = [col for col in dataset_columns if col not in mapped_columns]

        if unmapped:
            warnings.append({
                'type': 'unmapped_columns',
                'columns': unmapped,
                'message': f"{len(unmapped)} dataset column(s) will not be used: {', '.join(unmapped[:3])}"
            })

        # Calculate overall validity
        required_fields = [f for f in expected_fields.keys() if f not in optional_fields]
        valid = all(f in suggestions for f in required_fields)

        # Calculate compatibility score (0-100)
        compatibility = 0
        if required_fields:
            matched_required = sum(1 for f in required_fields if f in suggestions)
            compatibility = (matched_required / len(required_fields)) * 100

            # Boost score for high confidence matches
            if suggestions:
                avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                compatibility = (compatibility + avg_confidence * 100) / 2

        return jsonify({
            'valid': valid,
            'suggestions': suggestions,
            'confidence_scores': confidence_scores,
            'warnings': warnings,
            'compatibility_score': round(compatibility, 1),
            'expected_fields': expected_fields,
            'optional_fields': optional_fields,
            'field_examples': preset.field_examples
        })

    except Exception as e:
        logger.error(f"Error validating reward fields: {e}")
        return jsonify({'error': str(e)}), 500


# Custom preset management routes

@rewards_bp.route('/custom-presets', methods=['GET'])
def get_custom_presets():
    """Get all user-created custom presets."""
    try:
        return jsonify(reward_service.get_custom_presets())
    except Exception as e:
        logger.error(f"Failed to get custom presets: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/custom-preset', methods=['POST'])
def save_custom_preset():
    """Save a new custom reward preset."""
    try:
        data = request.get_json()
        name = data.get('name')
        preset_config = {
            'description': data.get('description', ''),
            'components': data.get('components', []),
            'example_input': data.get('example_input', ''),
            'example_output': data.get('example_output', ''),
            'difficulty': data.get('difficulty', 'intermediate'),
            'tags': data.get('tags', ['custom'])
        }

        if not name:
            return jsonify({'error': 'Preset name is required'}), 400

        if not preset_config['components']:
            return jsonify({'error': 'At least one component is required'}), 400

        success = reward_service.save_custom_preset(name, preset_config)

        if success:
            return jsonify({
                'success': True,
                'message': f'Custom preset "{name}" saved successfully',
                'preset_name': name
            })
        else:
            return jsonify({
                'error': f'Failed to save preset "{name}". It may already exist.'
            }), 400

    except Exception as e:
        logger.error(f"Failed to save custom preset: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/custom-preset/<preset_name>', methods=['DELETE'])
def delete_custom_preset(preset_name):
    """Delete a custom reward preset."""
    try:
        success = reward_service.delete_custom_preset(preset_name)

        if success:
            return jsonify({
                'success': True,
                'message': f'Custom preset "{preset_name}" deleted successfully'
            })
        else:
            return jsonify({
                'error': f'Preset "{preset_name}" not found or is not a custom preset.'
            }), 404

    except Exception as e:
        logger.error(f"Failed to delete custom preset: {e}")
        return jsonify({'error': str(e)}), 500


@rewards_bp.route('/custom-preset/<preset_name>', methods=['PUT'])
def update_custom_preset(preset_name):
    """Update an existing custom reward preset."""
    try:
        data = request.get_json()
        preset_config = {
            'description': data.get('description'),
            'components': data.get('components')
        }

        # Remove None values
        preset_config = {k: v for k, v in preset_config.items() if v is not None}

        success = reward_service.update_custom_preset(preset_name, preset_config)

        if success:
            return jsonify({
                'success': True,
                'message': f'Custom preset "{preset_name}" updated successfully'
            })
        else:
            return jsonify({
                'error': f'Preset "{preset_name}" not found or is not a custom preset.'
            }), 404

    except Exception as e:
        logger.error(f"Failed to update custom preset: {e}")
        return jsonify({'error': str(e)}), 500
