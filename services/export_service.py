"""
Export service for model export operations.

This module provides business logic for:
- Model export to various formats (HuggingFace, GGUF, etc.)
- Export listing and management
- Checkpoint management
- Batch export operations
"""

import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

from core import ModelExporter
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ExportService:
    """Service for managing model exports."""

    def __init__(self, outputs_folder: str = './outputs', exports_folder: str = './exports'):
        """
        Initialize export service.

        Args:
            outputs_folder: Path to training outputs directory
            exports_folder: Path to exports directory
        """
        self.outputs_folder = Path(outputs_folder)
        self.exports_folder = Path(exports_folder)

        # Ensure directories exist
        self.outputs_folder.mkdir(exist_ok=True)
        self.exports_folder.mkdir(exist_ok=True)

    def get_export_formats(self) -> Dict[str, Any]:
        """
        Get available export formats and quantization options.

        Returns:
            Dictionary with supported formats and quantizations
        """
        return {
            'formats': ModelExporter.SUPPORTED_FORMATS,
            'gguf_quantizations': ModelExporter.GGUF_QUANTIZATIONS
        }

    def list_exports(self, session_id: str) -> Dict[str, Any]:
        """
        List all exports for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with exports information
        """
        try:
            exporter = ModelExporter()
            return exporter.list_exports(session_id)
        except Exception as e:
            logger.error(f"Error listing exports for session {session_id}: {e}")
            raise

    def export_model(self, session_id: str, export_config: Dict[str, Any],
                    training_session=None, progress_callback: Optional[Callable] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Export a trained model to specified format.

        Args:
            session_id: Session identifier
            export_config: Export configuration with format, name, quantization, etc.
            training_session: Optional TrainingSession object if model is in memory
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success, export_path, metadata)

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If checkpoint doesn't exist
        """
        export_format = export_config.get('format', 'huggingface')
        export_name = export_config.get('name', None)
        quantization = export_config.get('quantization', 'q4_k_m')
        merge_lora = export_config.get('merge_lora', False)

        # Try to use trainer's export method if available and model is loaded
        if (training_session and
            hasattr(training_session, 'trainer') and
            training_session.trainer is not None and
            hasattr(training_session.trainer, 'model') and
            training_session.trainer.model is not None):

            logger.info(f"Exporting using in-memory model for session {session_id}")
            success, export_path, metadata = training_session.trainer.export_model(
                export_format=export_format,
                export_name=export_name,
                quantization=quantization if export_format == 'gguf' else None,
                merge_lora=merge_lora,
                progress_callback=progress_callback
            )
        else:
            # Export directly from checkpoint files
            logger.info(f"Exporting from checkpoint files for session {session_id}")

            # Find checkpoint
            checkpoint_path = self._find_checkpoint(session_id)
            if not checkpoint_path:
                raise FileNotFoundError(f'No checkpoint found for session {session_id}')

            # Use ModelExporter directly
            exporter = ModelExporter()
            success, export_path, metadata = exporter.export_model(
                model_path=str(checkpoint_path),
                session_id=session_id,
                export_format=export_format,
                export_name=export_name,
                quantization=quantization if export_format == 'gguf' else None,
                merge_lora=merge_lora,
                progress_callback=progress_callback
            )

        return success, export_path, metadata

    def _find_checkpoint(self, session_id: str) -> Optional[Path]:
        """
        Find the checkpoint for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to checkpoint, or None if not found
        """
        # Check for final checkpoint
        checkpoint_path = self.outputs_folder / session_id / "checkpoints" / "final"
        if checkpoint_path.exists():
            return checkpoint_path

        # Try to find any checkpoint
        checkpoints_dir = self.outputs_folder / session_id / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
            if checkpoints:
                # Use the most recent checkpoint
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using checkpoint: {checkpoint_path}")
                return checkpoint_path

        return None

    def delete_export(self, session_id: str, export_format: str, export_name: str) -> bool:
        """
        Delete a specific export.

        Args:
            session_id: Session identifier
            export_format: Export format (huggingface, gguf, etc.)
            export_name: Name of the export

        Returns:
            True if deleted successfully, False if not found

        Raises:
            Exception: If deletion fails
        """
        try:
            # Construct the export path
            export_path = self.exports_folder / session_id / export_format / export_name

            if not export_path.exists():
                return False

            # Delete the export (file or directory)
            if export_path.is_dir():
                shutil.rmtree(export_path)
            else:
                export_path.unlink()

            logger.info(f"Deleted export: {export_path}")

            # Check if this was the last export in the format directory
            format_dir = export_path.parent
            if format_dir.exists() and not any(format_dir.iterdir()):
                format_dir.rmdir()
                logger.info(f"Removed empty format directory: {format_dir}")

            # Check if this was the last export for the session
            session_dir = format_dir.parent
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
                logger.info(f"Removed empty session directory: {session_dir}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete export {export_name}: {e}")
            raise

    def list_checkpoints(self, session_id: str) -> Dict[str, Any]:
        """
        List available checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with checkpoint information

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        checkpoints_dir = self.outputs_folder / session_id / "checkpoints"

        if not checkpoints_dir.exists():
            raise FileNotFoundError(f'No checkpoints found for session {session_id}')

        checkpoints = []
        for checkpoint_path in checkpoints_dir.iterdir():
            if checkpoint_path.is_dir():
                # Get checkpoint metadata
                stat = checkpoint_path.stat()
                checkpoints.append({
                    'name': checkpoint_path.name,
                    'path': str(checkpoint_path.relative_to(self.outputs_folder)),
                    'size_bytes': sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file()),
                    'created': stat.st_ctime,
                    'modified': stat.st_mtime
                })

        # Sort by modified time (most recent first)
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)

        return {
            'session_id': session_id,
            'checkpoints': checkpoints,
            'count': len(checkpoints)
        }

    def batch_export(self, session_ids: List[str], export_format: str,
                    quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Export multiple models in batch.

        Args:
            session_ids: List of session identifiers
            export_format: Export format to use
            quantization: Optional quantization for GGUF format

        Returns:
            Dictionary with batch export results
        """
        if not session_ids:
            raise ValueError('No session IDs provided')

        results = []
        exporter = ModelExporter()

        for session_id in session_ids:
            # Check if model exists
            checkpoint_path = self._find_checkpoint(session_id)

            if not checkpoint_path:
                results.append({
                    'session_id': session_id,
                    'success': False,
                    'error': 'Model checkpoint not found'
                })
                continue

            # Export the model
            try:
                success, export_path, metadata = exporter.export_model(
                    model_path=str(checkpoint_path),
                    session_id=session_id,
                    export_format=export_format,
                    quantization=quantization if export_format == 'gguf' else None
                )

                results.append({
                    'session_id': session_id,
                    'success': success,
                    'export_path': export_path if success else None,
                    'metadata': metadata if success else None,
                    'error': metadata.get('error') if not success else None
                })
            except Exception as e:
                logger.error(f"Error exporting session {session_id}: {e}")
                results.append({
                    'session_id': session_id,
                    'success': False,
                    'error': str(e)
                })

        return {
            'results': results,
            'total': len(results),
            'successful': sum(1 for r in results if r['success'])
        }

    def delete_model(self, session_id: str, training_sessions: Dict) -> Tuple[bool, List[str]]:
        """
        Delete a model and all its associated data.

        Args:
            session_id: Session identifier
            training_sessions: Dictionary of active training sessions

        Returns:
            Tuple of (success, list of deleted items)

        Raises:
            ValueError: If model is currently training
            Exception: If deletion fails
        """
        # Check if model is currently training
        if session_id in training_sessions:
            session = training_sessions.get(session_id)
            if hasattr(session, 'status') and session.status == 'running':
                raise ValueError('Cannot delete model while training is in progress')

        deleted_items = []

        # Delete model outputs directory
        output_path = self.outputs_folder / session_id
        if output_path.exists():
            shutil.rmtree(output_path)
            deleted_items.append(f"outputs/{session_id}")
            logger.info(f"Deleted outputs for session {session_id}")

        # Delete exports directory
        export_path = self.exports_folder / session_id
        if export_path.exists():
            shutil.rmtree(export_path)
            deleted_items.append(f"exports/{session_id}")
            logger.info(f"Deleted exports for session {session_id}")

        # Remove from training sessions if exists
        if session_id in training_sessions:
            del training_sessions[session_id]
            deleted_items.append("session data")

        return True, deleted_items

    def get_export_path(self, session_id: str, export_path: str) -> Optional[Path]:
        """
        Get the full path to an export.

        Args:
            session_id: Session identifier
            export_path: Relative export path

        Returns:
            Full Path to export, or None if not found
        """
        full_path = self.exports_folder / session_id / export_path

        if not full_path.exists():
            return None

        return full_path

    def create_archive(self, export_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Create a zip archive of an export directory.

        Args:
            export_path: Path to export directory

        Returns:
            Tuple of (success, archive_path)
        """
        try:
            exporter = ModelExporter()
            return exporter.create_archive(str(export_path))
        except Exception as e:
            logger.error(f"Failed to create archive for {export_path}: {e}")
            return False, None
