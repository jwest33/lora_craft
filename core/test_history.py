"""
Test History Management System
Handles storage and retrieval of model test results
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Represents a single test result."""
    id: str
    timestamp: str
    prompt: str
    response: str
    model: str  # session_id or model_name
    parameters: Dict[str, Any]  # temperature, top_p, max_tokens, etc.
    metrics: Dict[str, Any]  # generation_time, tokens_per_second, token_count
    config: Optional[Dict[str, Any]] = None  # Additional configuration

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary."""
        return cls(**data)


class TestHistoryManager:
    """Manages test history storage and retrieval."""

    def __init__(self, base_dir: str = "./outputs/test_history"):
        """Initialize test history manager.

        Args:
            base_dir: Base directory for storing test history
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache = []  # In-memory cache of recent tests
        self._cache_size = 100
        self._load_recent_tests()

    def _get_monthly_file(self, date: Optional[datetime] = None) -> Path:
        """Get the monthly file path for storing tests.

        Args:
            date: Date to get file for (defaults to current date)

        Returns:
            Path to monthly JSON file
        """
        if date is None:
            date = datetime.now()

        filename = f"tests_{date.strftime('%Y-%m')}.json"
        return self.base_dir / filename

    def _load_recent_tests(self):
        """Load recent tests into cache."""
        try:
            # Load current month's tests
            current_file = self._get_monthly_file()
            if current_file.exists():
                with open(current_file, 'r') as f:
                    data = json.load(f)
                    self._cache = [TestResult.from_dict(test) for test in data.get('tests', [])]
                    # Keep only most recent tests in cache
                    self._cache = self._cache[-self._cache_size:]

            logger.info(f"Loaded {len(self._cache)} recent tests into cache")
        except Exception as e:
            logger.error(f"Error loading test history: {e}")
            self._cache = []

    def save_test(self, test_data: Dict[str, Any]) -> str:
        """Save a test result.

        Args:
            test_data: Test data including prompt, response, model, etc.

        Returns:
            Test ID
        """
        with self._lock:
            try:
                # Generate unique ID
                test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

                # Create TestResult
                test_result = TestResult(
                    id=test_id,
                    timestamp=datetime.now().isoformat(),
                    prompt=test_data.get('prompt', ''),
                    response=test_data.get('response', ''),
                    model=test_data.get('model', 'unknown'),
                    parameters=test_data.get('parameters', {}),
                    metrics=test_data.get('metrics', {}),
                    config=test_data.get('config')
                )

                # Add to cache
                self._cache.append(test_result)
                if len(self._cache) > self._cache_size:
                    self._cache.pop(0)

                # Save to file
                self._save_to_file(test_result)

                logger.info(f"Saved test {test_id}")
                return test_id

            except Exception as e:
                logger.error(f"Error saving test: {e}")
                raise

    def _save_to_file(self, test_result: TestResult):
        """Save test result to monthly file.

        Args:
            test_result: Test result to save
        """
        try:
            file_path = self._get_monthly_file()

            # Load existing data
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {'tests': [], 'metadata': {'created': datetime.now().isoformat()}}

            # Append new test
            data['tests'].append(test_result.to_dict())
            data['metadata']['last_updated'] = datetime.now().isoformat()
            data['metadata']['count'] = len(data['tests'])

            # Save back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving test to file: {e}")
            raise

    def get_test_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get test history.

        Args:
            limit: Maximum number of tests to return
            offset: Number of tests to skip

        Returns:
            List of test results
        """
        with self._lock:
            try:
                all_tests = []

                # Get all monthly files, sorted by date (newest first)
                files = sorted(self.base_dir.glob("tests_*.json"), reverse=True)

                for file_path in files:
                    if len(all_tests) >= limit + offset:
                        break

                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Add tests in reverse order (newest first)
                            tests = data.get('tests', [])
                            all_tests.extend(reversed(tests))
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                        continue

                # Apply offset and limit
                result = all_tests[offset:offset + limit]

                logger.info(f"Retrieved {len(result)} tests (offset={offset}, limit={limit})")
                return result

            except Exception as e:
                logger.error(f"Error getting test history: {e}")
                return []

    def get_test_by_id(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific test by ID.

        Args:
            test_id: Test ID to retrieve

        Returns:
            Test result or None if not found
        """
        # Check cache first
        for test in self._cache:
            if test.id == test_id:
                return test.to_dict()

        # Search in files
        for file_path in self.base_dir.glob("tests_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for test in data.get('tests', []):
                        if test.get('id') == test_id:
                            return test
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue

        return None

    def delete_old_tests(self, days: int = 30):
        """Delete tests older than specified days.

        Args:
            days: Number of days to keep tests
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)

            for file_path in self.base_dir.glob("tests_*.json"):
                # Check file modification time
                if file_path.stat().st_mtime < cutoff_date:
                    logger.info(f"Deleting old test file: {file_path}")
                    file_path.unlink()

        except Exception as e:
            logger.error(f"Error deleting old tests: {e}")

    def search_tests(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search tests by prompt or response content.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching test results
        """
        results = []
        query_lower = query.lower()

        for file_path in sorted(self.base_dir.glob("tests_*.json"), reverse=True):
            if len(results) >= limit:
                break

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for test in reversed(data.get('tests', [])):
                        if len(results) >= limit:
                            break

                        # Search in prompt and response
                        if (query_lower in test.get('prompt', '').lower() or
                            query_lower in test.get('response', '').lower()):
                            results.append(test)

            except Exception as e:
                logger.warning(f"Error searching file {file_path}: {e}")
                continue

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get test history statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_tests': 0,
            'models_tested': set(),
            'monthly_counts': {},
            'average_generation_time': 0,
            'total_tokens_generated': 0
        }

        total_time = 0
        time_count = 0

        for file_path in self.base_dir.glob("tests_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    tests = data.get('tests', [])

                    # Extract month from filename
                    month = file_path.stem.replace('tests_', '')
                    stats['monthly_counts'][month] = len(tests)
                    stats['total_tests'] += len(tests)

                    for test in tests:
                        # Track models
                        model = test.get('model')
                        if model:
                            stats['models_tested'].add(model)

                        # Track metrics
                        metrics = test.get('metrics', {})
                        if 'generation_time' in metrics:
                            total_time += metrics['generation_time']
                            time_count += 1
                        if 'token_count' in metrics:
                            stats['total_tokens_generated'] += metrics['token_count']

            except Exception as e:
                logger.warning(f"Error processing file {file_path} for statistics: {e}")
                continue

        # Convert set to list for JSON serialization
        stats['models_tested'] = list(stats['models_tested'])

        # Calculate average
        if time_count > 0:
            stats['average_generation_time'] = total_time / time_count

        return stats


# Global instance
_test_history_manager = None

def get_test_history_manager() -> TestHistoryManager:
    """Get or create the global test history manager instance."""
    global _test_history_manager
    if _test_history_manager is None:
        _test_history_manager = TestHistoryManager()
    return _test_history_manager