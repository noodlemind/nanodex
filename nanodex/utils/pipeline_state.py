"""
Pipeline state tracking utilities.

Tracks the progress of the nanodex pipeline including configuration,
analysis, data generation, training, and RAG indexing.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from enum import Enum


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineState:
    """
    Track and manage pipeline state.

    The pipeline consists of:
    1. Configuration
    2. Analysis
    3. Data Generation
    4. Training
    5. RAG Indexing
    """

    def __init__(self, project_path: str = "."):
        """Initialize pipeline state."""
        self.project_path = Path(project_path)
        self.state_file = self.project_path / ".nanodex_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from file or create new state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                return self._create_initial_state()
        return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial pipeline state."""
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "steps": {
                "configuration": {"status": StepStatus.PENDING.value, "data": {}},
                "analysis": {"status": StepStatus.PENDING.value, "data": {}},
                "data_generation": {"status": StepStatus.PENDING.value, "data": {}},
                "training": {"status": StepStatus.PENDING.value, "data": {}},
                "rag_indexing": {"status": StepStatus.PENDING.value, "data": {}},
            },
        }

    def save_state(self):
        """Save current state to file."""
        self.state["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_step_status(self, step_name: str) -> StepStatus:
        """Get status of a specific step."""
        if step_name in self.state["steps"]:
            status_value = self.state["steps"][step_name]["status"]
            return StepStatus(status_value)
        return StepStatus.PENDING

    def set_step_status(self, step_name: str, status: StepStatus, data: Optional[Dict] = None):
        """Set status of a specific step."""
        if step_name in self.state["steps"]:
            self.state["steps"][step_name]["status"] = status.value
            if data:
                self.state["steps"][step_name]["data"].update(data)
            self.save_state()

    def get_step_data(self, step_name: str) -> Dict:
        """Get metadata for a specific step."""
        if step_name in self.state["steps"]:
            return self.state["steps"][step_name]["data"]
        return {}

    def check_configuration(self) -> bool:
        """Check if configuration exists."""
        config_file = self.project_path / "config.yaml"
        if config_file.exists():
            self.set_step_status("configuration", StepStatus.COMPLETED)
            return True
        return False

    def check_analysis(self) -> bool:
        """Check if analysis has been completed."""
        # Look for analysis output or state
        status = self.get_step_status("analysis")
        return status == StepStatus.COMPLETED

    def check_data_generation(self) -> bool:
        """Check if training data has been generated."""
        # Common data output directories
        data_dirs = [
            self.project_path / "data" / "train.json",
            self.project_path / "data" / "training_data.json",
        ]
        for data_dir in data_dirs:
            if data_dir.exists():
                self.set_step_status("data_generation", StepStatus.COMPLETED)
                return True
        return False

    def check_training(self) -> bool:
        """Check if model training has been completed."""
        # Look for model checkpoints
        model_dirs = [
            self.project_path / "models" / "checkpoint-final",
            self.project_path / "models" / "final",
        ]
        for model_dir in model_dirs:
            if model_dir.exists():
                self.set_step_status("training", StepStatus.COMPLETED)
                return True
        return False

    def check_rag_indexing(self) -> bool:
        """Check if RAG index has been built."""
        index_path = self.project_path / "models" / "rag_index"
        if index_path.exists():
            self.set_step_status("rag_indexing", StepStatus.COMPLETED)
            return True
        return False

    def get_next_step(self) -> Optional[str]:
        """Get the next recommended step in the pipeline."""
        steps = [
            ("configuration", "nanodex init"),
            ("analysis", "nanodex analyze"),
            ("data_generation", "nanodex data generate --mode free"),
            ("training", "nanodex train"),
            ("rag_indexing", "nanodex rag index"),
        ]

        for step_name, command in steps:
            if self.get_step_status(step_name) != StepStatus.COMPLETED:
                return command
        return None

    def get_progress_percentage(self) -> float:
        """Calculate overall pipeline progress percentage."""
        total_steps = len(self.state["steps"])
        completed_steps = sum(
            1
            for step in self.state["steps"].values()
            if step["status"] == StepStatus.COMPLETED.value
        )
        return (completed_steps / total_steps) * 100
