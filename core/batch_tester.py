"""
Batch Testing System
Handles running multiple test prompts against models
"""

import os
import json
import csv
import uuid
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchTestConfig:
    """Configuration for a batch test."""
    batch_id: str
    model: str  # session_id or model_name
    prompts: List[str]
    parameters: Dict[str, Any]  # temperature, top_p, max_tokens, etc.
    output_dir: str
    callback: Optional[Callable] = None  # Progress callback

@dataclass
class BatchTestResult:
    """Result of a single test in a batch."""
    index: int
    prompt: str
    response: str
    success: bool
    generation_time: float
    token_count: int
    error: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class BatchTestStatus:
    """Status of a batch test."""
    batch_id: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    total: int
    completed: int
    successful: int
    failed: int
    start_time: str
    end_time: Optional[str] = None
    elapsed_time: Optional[float] = None
    average_time: Optional[float] = None
    results_file: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class BatchTestRunner:
    """Runs batch tests against models."""

    def __init__(self, base_dir: str = "./outputs/batch_tests"):
        """Initialize batch test runner.

        Args:
            base_dir: Base directory for storing batch test results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.active_tests = {}  # batch_id -> BatchTestStatus
        self.test_threads = {}  # batch_id -> Thread
        self._lock = threading.Lock()

    def start_batch_test(self, model: str, prompts_file: str, parameters: Dict[str, Any],
                         model_tester=None) -> str:
        """Start a batch test.

        Args:
            model: Model to test (session_id or model_name)
            prompts_file: Path to file containing prompts (CSV or JSON)
            parameters: Generation parameters
            model_tester: ModelTester instance to use

        Returns:
            Batch test ID
        """
        try:
            # Generate unique batch ID
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Load prompts from file
            prompts = self._load_prompts(prompts_file)

            if not prompts:
                raise ValueError("No prompts found in file")

            # Create output directory
            output_dir = self.base_dir / batch_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create batch test config
            config = BatchTestConfig(
                batch_id=batch_id,
                model=model,
                prompts=prompts,
                parameters=parameters,
                output_dir=str(output_dir)
            )

            # Create status
            status = BatchTestStatus(
                batch_id=batch_id,
                status='pending',
                total=len(prompts),
                completed=0,
                successful=0,
                failed=0,
                start_time=datetime.now().isoformat()
            )

            # Store in active tests
            with self._lock:
                self.active_tests[batch_id] = status

            # Start test thread
            thread = threading.Thread(
                target=self._run_batch_test,
                args=(config, status, model_tester)
            )
            thread.daemon = True
            thread.start()

            self.test_threads[batch_id] = thread

            logger.info(f"Started batch test {batch_id} with {len(prompts)} prompts")
            return batch_id

        except Exception as e:
            logger.error(f"Error starting batch test: {e}")
            raise

    def _load_prompts(self, prompts_file: str) -> List[str]:
        """Load prompts from file.

        Args:
            prompts_file: Path to prompts file

        Returns:
            List of prompts
        """
        prompts = []
        file_path = Path(prompts_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        prompts = data
                    elif isinstance(data, dict) and 'prompts' in data:
                        prompts = data['prompts']
                    else:
                        raise ValueError("Invalid JSON format")

            elif file_path.suffix == '.csv':
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Try common column names
                        prompt = row.get('prompt') or row.get('text') or row.get('input')
                        if prompt:
                            prompts.append(prompt)

            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]

            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise

        return prompts

    def _run_batch_test(self, config: BatchTestConfig, status: BatchTestStatus, model_tester):
        """Run the batch test (in thread).

        Args:
            config: Batch test configuration
            status: Batch test status to update
            model_tester: ModelTester instance
        """
        try:
            # Update status to running
            with self._lock:
                status.status = 'running'

            results = []
            start_time = time.time()

            # Process each prompt
            for i, prompt in enumerate(config.prompts):
                try:
                    # Check if cancelled
                    with self._lock:
                        if status.status == 'cancelled':
                            break

                    # Generate response
                    prompt_start = time.time()

                    # Use model tester if available
                    if model_tester:
                        response_data = model_tester.generate_response(
                            model_id=config.model,
                            prompt=prompt,
                            **config.parameters
                        )
                        response = response_data.get('response', '')
                        token_count = response_data.get('token_count', 0)
                    else:
                        # Fallback - would need actual model loading here
                        response = f"[Mock response to: {prompt[:50]}...]"
                        token_count = len(response.split())

                    generation_time = time.time() - prompt_start

                    # Create result
                    result = BatchTestResult(
                        index=i,
                        prompt=prompt,
                        response=response,
                        success=True,
                        generation_time=generation_time,
                        token_count=token_count
                    )

                    results.append(result)

                    # Update status
                    with self._lock:
                        status.completed += 1
                        status.successful += 1

                    logger.debug(f"Completed test {i+1}/{status.total}")

                except Exception as e:
                    logger.error(f"Error processing prompt {i}: {e}")

                    # Create error result
                    result = BatchTestResult(
                        index=i,
                        prompt=prompt,
                        response='',
                        success=False,
                        generation_time=0,
                        token_count=0,
                        error=str(e)
                    )

                    results.append(result)

                    # Update status
                    with self._lock:
                        status.completed += 1
                        status.failed += 1

                # Progress callback if provided
                if config.callback:
                    config.callback(status.to_dict())

            # Calculate statistics
            elapsed_time = time.time() - start_time
            successful_results = [r for r in results if r.success]

            if successful_results:
                avg_time = sum(r.generation_time for r in successful_results) / len(successful_results)
            else:
                avg_time = 0

            # Save results
            results_file = self._save_results(config, results, status)

            # Update final status
            with self._lock:
                if status.status != 'cancelled':
                    status.status = 'completed' if status.failed == 0 else 'completed_with_errors'
                status.end_time = datetime.now().isoformat()
                status.elapsed_time = elapsed_time
                status.average_time = avg_time
                status.results_file = str(results_file)

            logger.info(f"Batch test {config.batch_id} completed: {status.successful}/{status.total} successful")

        except Exception as e:
            logger.error(f"Batch test failed: {e}")

            with self._lock:
                status.status = 'failed'
                status.end_time = datetime.now().isoformat()
                status.error = str(e)

    def _save_results(self, config: BatchTestConfig, results: List[BatchTestResult],
                      status: BatchTestStatus) -> Path:
        """Save batch test results.

        Args:
            config: Batch test configuration
            results: List of test results
            status: Batch test status

        Returns:
            Path to results file
        """
        try:
            output_dir = Path(config.output_dir)

            # Save as JSON
            json_file = output_dir / 'results.json'
            with open(json_file, 'w') as f:
                json.dump({
                    'batch_id': config.batch_id,
                    'model': config.model,
                    'parameters': config.parameters,
                    'status': status.to_dict(),
                    'results': [r.to_dict() for r in results]
                }, f, indent=2)

            # Also save as CSV for easy analysis
            csv_file = output_dir / 'results.csv'
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'index', 'prompt', 'response', 'success',
                    'generation_time', 'token_count', 'error'
                ])
                writer.writeheader()
                for result in results:
                    writer.writerow(result.to_dict())

            logger.info(f"Saved batch test results to {output_dir}")
            return json_file

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def get_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch test.

        Args:
            batch_id: Batch test ID

        Returns:
            Status dictionary or None if not found
        """
        with self._lock:
            status = self.active_tests.get(batch_id)
            return status.to_dict() if status else None

    def get_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed batch test.

        Args:
            batch_id: Batch test ID

        Returns:
            Results dictionary or None if not found
        """
        try:
            results_file = self.base_dir / batch_id / 'results.json'

            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)

            return None

        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

    def cancel_test(self, batch_id: str) -> bool:
        """Cancel a running batch test.

        Args:
            batch_id: Batch test ID

        Returns:
            True if cancelled successfully
        """
        with self._lock:
            if batch_id in self.active_tests:
                self.active_tests[batch_id].status = 'cancelled'
                logger.info(f"Cancelled batch test {batch_id}")
                return True
            return False

    def get_active_test(self) -> Optional[Dict[str, Any]]:
        """Get the currently active test (if any).

        Returns:
            Active test status or None
        """
        with self._lock:
            for batch_id, status in self.active_tests.items():
                if status.status in ['pending', 'running']:
                    return status.to_dict()
            return None

    def list_batch_tests(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent batch tests.

        Args:
            limit: Maximum number of tests to return

        Returns:
            List of batch test summaries
        """
        tests = []

        try:
            # Get all batch test directories
            for test_dir in sorted(self.base_dir.glob("batch_*"), reverse=True)[:limit]:
                results_file = test_dir / 'results.json'

                if results_file.exists():
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        tests.append({
                            'batch_id': data['batch_id'],
                            'model': data['model'],
                            'status': data['status'],
                            'total': data['status']['total'],
                            'successful': data['status']['successful'],
                            'failed': data['status']['failed']
                        })

        except Exception as e:
            logger.error(f"Error listing batch tests: {e}")

        return tests


# Global instance
_batch_test_runner = None

def get_batch_test_runner() -> BatchTestRunner:
    """Get or create the global batch test runner instance."""
    global _batch_test_runner
    if _batch_test_runner is None:
        _batch_test_runner = BatchTestRunner()
    return _batch_test_runner