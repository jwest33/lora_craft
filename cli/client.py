"""
API Client for LoRA Craft

Handles all HTTP communication with the LoRA Craft Flask backend.
"""

import json
import requests
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urljoin
import time


class APIClient:
    """Client for interacting with LoRA Craft API."""

    def __init__(self, base_url: str = "http://localhost:5001", timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the LoRA Craft server
            timeout: Default request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        """Construct full URL from path."""
        return urljoin(self.base_url + '/', path.lstrip('/'))

    def _handle_response(self, response: requests.Response) -> Tuple[bool, Any]:
        """
        Handle API response and extract data.

        Returns:
            Tuple of (success, data/error_message)
        """
        try:
            response.raise_for_status()
            data = response.json()

            # Check for success field in response
            if isinstance(data, dict) and 'error' in data:
                return False, data['error']

            return True, data

        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', str(e))
            except:
                error_msg = str(e)
            return False, error_msg

        except requests.exceptions.RequestException as e:
            return False, f"Request failed: {str(e)}"

        except json.JSONDecodeError:
            return False, "Invalid JSON response from server"

    # ==================== System Endpoints ====================

    def get_system_info(self) -> Tuple[bool, Any]:
        """Get system information."""
        response = self.session.get(self._url('/api/system/info'), timeout=self.timeout)
        return self._handle_response(response)

    def get_system_status(self) -> Tuple[bool, Any]:
        """Get real-time system status."""
        response = self.session.get(self._url('/api/system_status'), timeout=self.timeout)
        return self._handle_response(response)

    def health_check(self) -> Tuple[bool, Any]:
        """Check if server is healthy."""
        response = self.session.get(self._url('/api/system/health'), timeout=5)
        return self._handle_response(response)

    # ==================== Dataset Endpoints ====================

    def list_datasets(self) -> Tuple[bool, Any]:
        """List available datasets."""
        response = self.session.get(self._url('/api/datasets/list'), timeout=self.timeout)
        return self._handle_response(response)

    def get_dataset_status(self, dataset_name: str, dataset_config: Optional[str] = None) -> Tuple[bool, Any]:
        """Get dataset cache status."""
        params = {'dataset_config': dataset_config} if dataset_config else {}
        safe_name = dataset_name.replace('/', '__')
        response = self.session.get(
            self._url(f'/api/datasets/status/{safe_name}'),
            params=params,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def download_dataset(self, dataset_name: str, dataset_config: Optional[str] = None,
                        force_download: bool = False) -> Tuple[bool, Any]:
        """Download a HuggingFace dataset."""
        data = {
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'force_download': force_download
        }
        response = self.session.post(
            self._url('/api/datasets/download'),
            json=data,
            timeout=300  # 5 minute timeout for downloads
        )
        return self._handle_response(response)

    def sample_dataset(self, dataset_name: str, dataset_config: Optional[str] = None,
                      sample_size: int = 5) -> Tuple[bool, Any]:
        """Get samples from a dataset."""
        data = {
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'sample_size': sample_size
        }
        response = self.session.post(
            self._url('/api/datasets/sample'),
            json=data,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def detect_dataset_fields(self, dataset_name: str, dataset_config: Optional[str] = None,
                             is_local: bool = False) -> Tuple[bool, Any]:
        """Detect fields in a dataset."""
        data = {
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'is_local': is_local
        }
        response = self.session.post(
            self._url('/api/datasets/detect-fields'),
            json=data,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def upload_dataset(self, file_path: str) -> Tuple[bool, Any]:
        """Upload a dataset file."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                self._url('/api/datasets/upload'),
                files=files,
                timeout=300  # 5 minute timeout for uploads
            )
        return self._handle_response(response)

    def list_uploaded_datasets(self) -> Tuple[bool, Any]:
        """List uploaded datasets."""
        response = self.session.get(self._url('/api/datasets/uploaded'), timeout=self.timeout)
        return self._handle_response(response)

    def clear_dataset_cache(self) -> Tuple[bool, Any]:
        """Clear dataset cache."""
        response = self.session.post(self._url('/api/datasets/cache/clear'), timeout=self.timeout)
        return self._handle_response(response)

    # ==================== Training Endpoints ====================

    def start_training(self, config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Start a training session."""
        response = self.session.post(
            self._url('/api/training/start'),
            json=config,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def get_training_status(self, session_id: str) -> Tuple[bool, Any]:
        """Get training session status."""
        response = self.session.get(
            self._url(f'/api/training/{session_id}/status'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def stop_training(self, session_id: str) -> Tuple[bool, Any]:
        """Stop a training session."""
        response = self.session.post(
            self._url(f'/api/training/{session_id}/stop'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def get_training_metrics(self, session_id: str) -> Tuple[bool, Any]:
        """Get training metrics."""
        response = self.session.get(
            self._url(f'/api/training/{session_id}/metrics'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def get_training_logs(self, session_id: str, limit: int = 100) -> Tuple[bool, Any]:
        """Get training logs."""
        params = {'limit': limit}
        response = self.session.get(
            self._url(f'/api/training/{session_id}/logs'),
            params=params,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def list_training_sessions(self) -> Tuple[bool, Any]:
        """List all training sessions."""
        response = self.session.get(self._url('/api/training/sessions'), timeout=self.timeout)
        return self._handle_response(response)

    def get_training_history(self, session_id: str) -> Tuple[bool, Any]:
        """Get training history for reconnection."""
        response = self.session.get(
            self._url(f'/api/training/session/{session_id}/history'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    # ==================== Model Endpoints ====================

    def list_available_models(self) -> Tuple[bool, Any]:
        """List available base models."""
        response = self.session.get(self._url('/api/models'), timeout=self.timeout)
        return self._handle_response(response)

    def list_trained_models(self) -> Tuple[bool, Any]:
        """List trained models."""
        response = self.session.get(self._url('/api/models/trained'), timeout=self.timeout)
        return self._handle_response(response)

    def get_model_info(self, session_id: str) -> Tuple[bool, Any]:
        """Get model information."""
        response = self.session.get(
            self._url(f'/api/models/{session_id}/info'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def load_model(self, model_id: str, model_type: str = 'trained') -> Tuple[bool, Any]:
        """Load a model for testing."""
        data = {
            'model_id': model_id,
            'type': model_type
        }
        response = self.session.post(
            self._url('/api/test/load'),
            json=data,
            timeout=60  # Model loading can take time
        )
        return self._handle_response(response)

    def generate_response(self, model_id: str, prompt: str,
                         config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """Generate a response from a loaded model."""
        data = {
            'model_id': model_id,
            'prompt': prompt,
            'config': config or {}
        }
        response = self.session.post(
            self._url('/api/test/generate'),
            json=data,
            timeout=120  # Generation can take time
        )
        return self._handle_response(response)

    def compare_models(self, model_ids: List[str], prompt: str,
                      config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """Compare multiple models."""
        data = {
            'model_ids': model_ids,
            'prompt': prompt,
            'config': config or {}
        }
        response = self.session.post(
            self._url('/api/test/compare'),
            json=data,
            timeout=300  # Multiple models take longer
        )
        return self._handle_response(response)

    # ==================== Export Endpoints ====================

    def get_export_formats(self) -> Tuple[bool, Any]:
        """Get available export formats."""
        response = self.session.get(self._url('/api/export/formats'), timeout=self.timeout)
        return self._handle_response(response)

    def list_exports(self, session_id: str) -> Tuple[bool, Any]:
        """List exports for a session."""
        response = self.session.get(
            self._url(f'/api/export/list/{session_id}'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def export_model(self, session_id: str, export_config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Export a trained model."""
        response = self.session.post(
            self._url(f'/api/export/{session_id}'),
            json=export_config,
            timeout=600  # Export can take up to 10 minutes
        )
        return self._handle_response(response)

    def delete_model(self, session_id: str) -> Tuple[bool, Any]:
        """Delete a model and all associated data."""
        response = self.session.delete(
            self._url(f'/api/models/{session_id}'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    # ==================== Configuration Endpoints ====================

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate a training configuration."""
        response = self.session.post(
            self._url('/api/config/validate'),
            json=config,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def save_config(self, config: Dict[str, Any], filename: Optional[str] = None) -> Tuple[bool, Any]:
        """Save a configuration."""
        data = config.copy()
        if filename:
            data['filename'] = filename
        response = self.session.post(
            self._url('/api/config/save'),
            json=data,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def load_config(self, filename: str) -> Tuple[bool, Any]:
        """Load a saved configuration."""
        response = self.session.get(
            self._url(f'/api/config/load/{filename}'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def list_configs(self) -> Tuple[bool, Any]:
        """List saved configurations."""
        response = self.session.get(self._url('/api/configs/list'), timeout=self.timeout)
        return self._handle_response(response)

    # ==================== Reward Endpoints ====================

    def list_reward_presets(self) -> Tuple[bool, Any]:
        """List available reward presets."""
        response = self.session.get(self._url('/api/rewards/presets'), timeout=self.timeout)
        return self._handle_response(response)

    def get_reward_preset_details(self, preset_name: str) -> Tuple[bool, Any]:
        """Get details of a reward preset."""
        response = self.session.get(
            self._url(f'/api/rewards/preset-details/{preset_name}'),
            timeout=self.timeout
        )
        return self._handle_response(response)

    def test_reward(self, reward_config: Dict[str, Any],
                   test_cases: List[Dict[str, Any]]) -> Tuple[bool, Any]:
        """Test a reward configuration."""
        data = {
            'reward_config': reward_config,
            'test_cases': test_cases
        }
        response = self.session.post(
            self._url('/api/rewards/test'),
            json=data,
            timeout=self.timeout
        )
        return self._handle_response(response)

    def validate_reward_fields(self, preset_name: str,
                              dataset_columns: List[str]) -> Tuple[bool, Any]:
        """Validate reward field mappings for a dataset."""
        data = {
            'reward_preset': preset_name,
            'dataset_columns': dataset_columns
        }
        response = self.session.post(
            self._url('/api/rewards/validate-fields'),
            json=data,
            timeout=self.timeout
        )
        return self._handle_response(response)
