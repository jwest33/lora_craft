"""
Export routes for model export and checkpoint management.

This module provides endpoints for:
- Exporting models to various formats (HuggingFace, GGUF, etc.)
- Listing and deleting exports
- Downloading exported models
- Managing checkpoints
- Batch export operations
"""

from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, current_app

from services import ExportService
from websockets.utils import emit_to_session
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Create blueprint
exports_bp = Blueprint('exports', __name__, url_prefix='/api')

# Initialize service
export_service = ExportService()


@exports_bp.route('/export/formats', methods=['GET'])
def get_export_formats():
    """Get available export formats and quantization options."""
    try:
        return jsonify(export_service.get_export_formats())
    except Exception as e:
        logger.error(f"Error getting export formats: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/export/list/<session_id>', methods=['GET'])
def list_exports(session_id):
    """List all exports for a session."""
    try:
        exports = export_service.list_exports(session_id)
        return jsonify(exports)
    except Exception as e:
        logger.error(f"Error listing exports: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/export/<session_id>', methods=['POST'])
def export_model(session_id):
    """Export trained model."""
    training_sessions = current_app.training_sessions

    # First check if session exists in memory
    session_obj = training_sessions.get(session_id)
    if session_obj and session_obj.status != 'completed':
        return jsonify({'error': 'Training not completed'}), 400

    try:
        export_config = request.json or {}

        # Progress callback for WebSocket updates
        def progress_callback(message, progress):
            emit_to_session(current_app.socketio, session_id, 'export_progress', {
                'message': message,
                'progress': progress
            })

        # Use the service to export
        success, export_path, metadata = export_service.export_model(
            session_id=session_id,
            export_config=export_config,
            training_session=session_obj,
            progress_callback=progress_callback
        )

        if success:
            return jsonify({
                'success': True,
                'path': export_path,
                'format': export_config.get('format', 'huggingface'),
                'metadata': metadata
            })
        else:
            return jsonify({
                'success': False,
                'error': metadata.get('error', 'Export failed')
            }), 500

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/export_model', methods=['POST'])
def export_model_generic():
    """Export a model to specified format (generic endpoint)."""
    try:
        data = request.json
        model_path = data.get('model_path')
        export_format = data.get('format', 'safetensors')
        output_name = data.get('output_name', 'exported_model')

        if not model_path:
            return jsonify({'error': 'Model path required', 'success': False}), 400

        # Determine session ID from path
        session_id = Path(model_path).name if Path(model_path).exists() else model_path

        # Export model using service
        export_config = {
            'format': export_format,
            'name': output_name
        }

        success, export_path, metadata = export_service.export_model(
            session_id=session_id,
            export_config=export_config
        )

        if success:
            return jsonify({
                'success': True,
                'format': export_format,
                'output_path': export_path,
                'output_name': output_name,
                'download_url': f"/api/export/download/{session_id}/{export_format}/{output_name}",
                'metadata': metadata
            })
        else:
            return jsonify({
                'success': False,
                'error': metadata.get('error', 'Export failed')
            }), 500

    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@exports_bp.route('/export/download/<session_id>/<path:export_path>', methods=['GET'])
def download_export(session_id, export_path):
    """Download an exported model."""
    try:
        # Get the full path to the export
        full_path = export_service.get_export_path(session_id, export_path)

        if not full_path:
            return jsonify({'error': 'Export not found'}), 404

        # If it's a directory, create a zip
        if full_path.is_dir():
            success, archive_path = export_service.create_archive(full_path)
            if success:
                return send_file(
                    archive_path,
                    as_attachment=True,
                    download_name=f"{full_path.name}.zip"
                )
            else:
                return jsonify({'error': 'Failed to create archive'}), 500
        else:
            # Send single file
            return send_file(
                str(full_path),
                as_attachment=True,
                download_name=full_path.name
            )

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/export/<session_id>/<export_format>/<export_name>', methods=['DELETE'])
def delete_export(session_id, export_format, export_name):
    """Delete a specific export."""
    try:
        success = export_service.delete_export(session_id, export_format, export_name)

        if not success:
            return jsonify({'error': 'Export not found'}), 404

        return jsonify({
            'success': True,
            'message': f'Export {export_name} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Failed to delete export {export_name}: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/export/checkpoints/<session_id>', methods=['GET'])
def list_checkpoints(session_id):
    """List available checkpoints for a session."""
    try:
        result = export_service.list_checkpoints(session_id)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'No checkpoints found for this session'}), 404
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/exports/batch', methods=['POST'])
def batch_export():
    """Export multiple models in batch."""
    try:
        data = request.json
        session_ids = data.get('session_ids', [])
        export_format = data.get('format', 'huggingface')
        quantization = data.get('quantization', 'q4_k_m')

        result = export_service.batch_export(
            session_ids=session_ids,
            export_format=export_format,
            quantization=quantization
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Batch export error: {e}")
        return jsonify({'error': str(e)}), 500


@exports_bp.route('/models/<session_id>', methods=['DELETE'])
def delete_model(session_id):
    """Delete a model and all its associated data."""
    try:
        training_sessions = current_app.training_sessions

        success, deleted_items = export_service.delete_model(session_id, training_sessions)

        return jsonify({
            'success': success,
            'message': f'Model {session_id} deleted successfully',
            'deleted': deleted_items
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to delete model {session_id}: {e}")
        return jsonify({'error': str(e)}), 500
