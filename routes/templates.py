"""
Template routes for prompt and chat template management.

This module provides endpoints for:
- Chat template management (Jinja2 templates)
- Prompt template management (GRPO templates)
- Template validation and preview
- Custom template CRUD operations
"""

import os
import json
from pathlib import Path
from flask import Blueprint, request, jsonify
from jinja2 import Environment, TemplateSyntaxError

from core import PromptTemplate, TemplateConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create blueprint
templates_bp = Blueprint('templates', __name__, url_prefix='/api/templates')


def get_configs_dir():
    """Get the configs directory path."""
    return Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs'))


# Chat template routes

@templates_bp.route('/chat-templates', methods=['GET'])
def get_chat_templates():
    """Get available chat templates."""
    try:
        # Built-in chat templates
        builtin_templates = {
            'grpo': {
                'name': 'GRPO Default',
                'template': "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ system_prompt + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ reasoning_start }}{% endif %}",
                'description': 'GRPO template optimized for reasoning tasks'
            }
        }

        # Load custom templates from file storage
        custom_templates = {}
        chat_templates_dir = Path('./chat_templates')
        if chat_templates_dir.exists():
            for template_file in chat_templates_dir.glob('*.json'):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        custom_templates[template_file.stem] = template_data
                except Exception as e:
                    logger.warning(f"Failed to load chat template {template_file}: {e}")

        return jsonify({
            'builtin': builtin_templates,
            'custom': custom_templates
        })
    except Exception as e:
        logger.error(f"Error getting chat templates: {e}")
        return jsonify({'error': str(e)}), 500


@templates_bp.route('/chat-template/save', methods=['POST'])
def save_chat_template():
    """Save a custom chat template."""
    try:
        data = request.json
        name = data.get('name')
        template = data.get('template')
        description = data.get('description', '')

        if not name or not template:
            return jsonify({'error': 'Name and template required'}), 400

        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        # Save template to file
        configs_dir = get_configs_dir()
        configs_dir.mkdir(exist_ok=True)

        template_path = configs_dir / f"chat_template_{safe_name}.json"
        with open(template_path, 'w') as f:
            json.dump({
                'name': name,
                'template': template,
                'description': description
            }, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Chat template "{name}" saved successfully',
            'template_id': safe_name
        })
    except Exception as e:
        logger.error(f"Error saving chat template: {e}")
        return jsonify({'error': str(e)}), 500


@templates_bp.route('/chat-template/validate', methods=['POST'])
def validate_chat_template():
    """Validate a Jinja2 chat template."""
    try:
        data = request.json
        template = data.get('template')

        if not template:
            return jsonify({'valid': False, 'error': 'No template provided'}), 400

        # Try to validate with Jinja2
        env = Environment()

        try:
            env.from_string(template)
            return jsonify({'valid': True, 'message': 'Template is valid'})
        except TemplateSyntaxError as e:
            return jsonify({'valid': False, 'error': str(e)})

    except Exception as e:
        logger.error(f"Error validating chat template: {e}")
        return jsonify({'error': str(e)}), 500


@templates_bp.route('/chat-template/preview', methods=['POST'])
def preview_chat_template():
    """Preview a chat template with sample data."""
    try:
        data = request.json
        template = data.get('template')
        messages = data.get('messages', [])
        system_prompt = data.get('system_prompt', '')
        reasoning_start = data.get('reasoning_start', '<start_working_out>')
        eos_token = data.get('eos_token', '</s>')
        add_generation_prompt = data.get('add_generation_prompt', False)

        if not template:
            return jsonify({'error': 'No template provided'}), 400

        # Render template
        env = Environment()

        try:
            tmpl = env.from_string(template)
            preview = tmpl.render(
                messages=messages,
                system_prompt=system_prompt,
                reasoning_start=reasoning_start,
                eos_token=eos_token,
                add_generation_prompt=add_generation_prompt
            )
            return jsonify({'preview': preview})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Error previewing chat template: {e}")
        return jsonify({'error': str(e)}), 500


# Prompt template routes

@templates_bp.route('', methods=['GET'])
def get_templates():
    """Get available prompt templates."""
    try:
        # Default GRPO templates
        default_templates = {
            'grpo-default': {
                'name': 'GRPO Default',
                'description': 'Default GRPO template for reasoning tasks',
                'reasoning_start': '<start_working_out>',
                'reasoning_end': '<end_working_out>',
                'solution_start': '<SOLUTION>',
                'solution_end': '</SOLUTION>',
                'system_prompt': 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>'
            },
            'qwen': {
                'name': 'Qwen GRPO',
                'description': 'GRPO template optimized for Qwen models',
                'reasoning_start': '<start_working_out>',
                'reasoning_end': '<end_working_out>',
                'solution_start': '<SOLUTION>',
                'solution_end': '</SOLUTION>',
                'system_prompt': 'You are given a problem to solve.\nProvide your reasoning within <start_working_out> and <end_working_out> tags.\nThen provide the final answer within <SOLUTION></SOLUTION> tags.'
            },
            'llama': {
                'name': 'LLaMA GRPO',
                'description': 'GRPO template optimized for LLaMA models',
                'reasoning_start': '[THINKING]',
                'reasoning_end': '[/THINKING]',
                'solution_start': '[ANSWER]',
                'solution_end': '[/ANSWER]',
                'system_prompt': 'You are a helpful assistant.\nShow your work within [THINKING] and [/THINKING].\nProvide the final answer within [ANSWER] and [/ANSWER].'
            }
        }

        # Load custom templates from file storage
        custom_templates = {}
        configs_dir = get_configs_dir()
        if configs_dir.exists():
            # Look for template files (those with template_ prefix)
            for template_file in configs_dir.glob('template_*.json'):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        # Remove the 'template_' prefix from the key
                        template_key = template_file.stem.replace('template_', '')
                        custom_templates[template_key] = template_data
                except Exception as e:
                    logger.warning(f"Failed to load template {template_file}: {e}")

        return jsonify({
            'default': default_templates,
            'custom': custom_templates
        })
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        return jsonify({'error': str(e)}), 500


@templates_bp.route('/save', methods=['POST'])
def save_template():
    """Save a custom prompt template."""
    try:
        template_data = request.json
        template_name = template_data.get('name')

        if not template_name:
            return jsonify({'error': 'Template name required'}), 400

        # Sanitize template name for filesystem
        safe_name = "".join(c for c in template_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        # Save template to file
        configs_dir = get_configs_dir()
        configs_dir.mkdir(exist_ok=True)

        template_path = configs_dir / f"template_{safe_name}.json"
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Template "{template_name}" saved successfully',
            'template_id': safe_name
        })
    except Exception as e:
        logger.error(f"Error saving template: {e}")
        return jsonify({'error': str(e)}), 500


@templates_bp.route('/<template_id>', methods=['DELETE'])
def delete_template(template_id):
    """Delete a custom template."""
    try:
        template_path = Path(f'./templates/{template_id}.json')
        if template_path.exists():
            template_path.unlink()
            return jsonify({'success': True, 'message': 'Template deleted'})
        else:
            return jsonify({'error': 'Template not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        return jsonify({'error': str(e)}), 500


@templates_bp.route('/test', methods=['POST'])
def test_template():
    """Test a template with sample data."""
    try:
        config = request.json

        # Create template config
        template_config = TemplateConfig(
            name="test",
            description="Test template",
            reasoning_start_marker=config.get('reasoning_start', '<start_working_out>'),
            reasoning_end_marker=config.get('reasoning_end', '<end_working_out>'),
            solution_start_marker=config.get('solution_start', '<SOLUTION>'),
            solution_end_marker=config.get('solution_end', '</SOLUTION>'),
            system_prompt=config.get('system_prompt')
        )

        template = PromptTemplate(template_config)

        # Test with sample data
        sample_instruction = config.get('sample_instruction', 'What is 2 + 2?')
        sample_response = config.get('sample_response', '2 + 2 = 4')

        # Format the sample
        messages = [
            {'role': 'system', 'content': template_config.system_prompt} if template_config.system_prompt else None,
            {'role': 'user', 'content': sample_instruction},
            {'role': 'assistant', 'content': sample_response}
        ]
        messages = [m for m in messages if m]  # Remove None entries

        # Generate preview
        preview = f"System: {template_config.system_prompt}\n\n" if template_config.system_prompt else ""
        preview += f"User: {sample_instruction}\n\n"
        preview += f"Assistant: {template_config.reasoning_start_marker}\n"
        preview += f"[Reasoning would go here]\n"
        preview += f"{template_config.reasoning_end_marker}\n"
        preview += f"{template_config.solution_start_marker}\n"
        preview += f"{sample_response}\n"
        preview += f"{template_config.solution_end_marker}"

        return jsonify({
            'success': True,
            'preview': preview,
            'formatted': preview
        })
    except Exception as e:
        logger.error(f"Error testing template: {e}")
        return jsonify({'error': str(e)}), 500
