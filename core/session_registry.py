"""Session Registry for tracking training sessions and models."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a training session."""
    session_id: str
    model_name: str
    status: str  # 'running', 'completed', 'error'
    checkpoint_path: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    best_reward: Optional[float] = None
    epochs_trained: Optional[int] = None
    training_config: Optional[Dict[str, Any]] = None
    exports: Optional[List[Dict[str, str]]] = None
    display_name: Optional[str] = None  # User-friendly name for the model

    def to_dict(self) -> Dict:
        """Convert to dictionary, handling None values."""
        data = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in data.items() if v is not None}


class SessionRegistry:
    """Manages a registry of training sessions for fast lookup."""

    def __init__(self, registry_path: str = "./outputs/sessions.json"):
        """Initialize the session registry.

        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

        # Load existing registry
        self.load()

    def load(self) -> None:
        """Load the registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for session_data in data.get('sessions', []):
                        session = SessionInfo(**session_data)
                        self.sessions[session.session_id] = session
                logger.info(f"Loaded {len(self.sessions)} sessions from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.sessions = {}

    def save(self) -> None:
        """Save the registry to disk."""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(exist_ok=True)

            with self._lock:
                data = {
                    'sessions': [session.to_dict() for session in self.sessions.values()],
                    'updated_at': datetime.now().isoformat()
                }

                # Write to temp file first for atomicity
                temp_path = self.registry_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)

                # Move temp file to final location
                temp_path.replace(self.registry_path)

            logger.debug(f"Saved {len(self.sessions)} sessions to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def add_session(self, session_info: SessionInfo) -> None:
        """Add or update a session in the registry.

        Args:
            session_info: Session information to add
        """
        with self._lock:
            self.sessions[session_info.session_id] = session_info
        self.save()

    def update_session(self, session_id: str, **updates) -> None:
        """Update specific fields of a session.

        Args:
            session_id: ID of the session to update
            **updates: Fields to update
        """
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                self.save()

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get a session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionInfo if found, None otherwise
        """
        return self.sessions.get(session_id)

    def get_checkpoint_path(self, session_id: str) -> Optional[str]:
        """Get the checkpoint path for a session.

        Args:
            session_id: Session ID to get checkpoint for

        Returns:
            Path to checkpoint if found, None otherwise
        """
        session = self.get_session(session_id)
        if session and session.checkpoint_path:
            return session.checkpoint_path

        # Fallback: try to find checkpoint in outputs directory
        checkpoint_path = Path(f"./outputs/{session_id}/checkpoints/final")
        if checkpoint_path.exists():
            return str(checkpoint_path)

        return None

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from the registry.

        Args:
            session_id: Session ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self.save()
                return True
        return False

    def list_completed_sessions(self) -> List[SessionInfo]:
        """List all completed sessions.

        Returns:
            List of completed sessions sorted by completion time
        """
        completed = [s for s in self.sessions.values() if s.status == 'completed']
        # Sort by completed_at, most recent first
        completed.sort(key=lambda s: s.completed_at or '', reverse=True)
        return completed

    def rebuild_from_directories(self, outputs_dir: str = "./outputs") -> int:
        """Rebuild the registry by scanning output directories.

        Args:
            outputs_dir: Path to outputs directory

        Returns:
            Number of sessions found
        """
        outputs_path = Path(outputs_dir)
        if not outputs_path.exists():
            return 0

        sessions_found = 0

        for session_dir in outputs_path.iterdir():
            if not session_dir.is_dir():
                continue

            # Check for checkpoints
            checkpoint_dir = session_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue

            # Look for final checkpoint
            final_checkpoint = checkpoint_dir / "final"
            if not final_checkpoint.exists():
                continue

            # Read training state if available
            state_file = final_checkpoint / "training_state.json"
            session_info = SessionInfo(
                session_id=session_dir.name,
                model_name="Unknown",
                status="completed",
                checkpoint_path=str(final_checkpoint),
                created_at=datetime.fromtimestamp(session_dir.stat().st_ctime).isoformat(),
                completed_at=datetime.fromtimestamp(final_checkpoint.stat().st_mtime).isoformat()
            )

            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        session_info.model_name = state.get('config', {}).get('model_name', 'Unknown')
                        session_info.best_reward = state.get('best_reward')
                        session_info.epochs_trained = state.get('current_epoch')
                        # Load training config for prompt formatting
                        session_info.training_config = state.get('config')
                except Exception as e:
                    logger.warning(f"Failed to read state for {session_dir.name}: {e}")

            # If training_config not found, try loading from config.json as fallback
            if not session_info.training_config:
                config_file = final_checkpoint / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            session_info.training_config = json.load(f)
                            logger.debug(f"Loaded training config from config.json for {session_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to read config.json for {session_dir.name}: {e}")

            self.add_session(session_info)
            sessions_found += 1

        logger.info(f"Rebuilt registry with {sessions_found} sessions")
        return sessions_found

    def cleanup_invalid_sessions(self) -> int:
        """Remove sessions that no longer have valid checkpoints.

        Returns:
            Number of sessions removed
        """
        removed = 0
        with self._lock:
            to_remove = []
            for session_id, session in self.sessions.items():
                if session.checkpoint_path:
                    if not Path(session.checkpoint_path).exists():
                        to_remove.append(session_id)

            for session_id in to_remove:
                del self.sessions[session_id]
                removed += 1

        if removed > 0:
            self.save()
            logger.info(f"Removed {removed} invalid sessions from registry")

        return removed
