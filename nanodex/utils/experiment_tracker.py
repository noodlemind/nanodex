"""
Experiment tracking for managing and comparing training runs.

Tracks hyperparameters, metrics, and results across multiple experiments
to help understand what configurations work best.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import os
from datetime import datetime
from dataclasses import dataclass
from rich.console import Console

console = Console()


def atomic_write_json(file_path: Path, data: Any) -> None:
    """
    Atomically write JSON data to a file.

    Args:
        file_path: Target file path
        data: Data to write (must be JSON serializable)

    Uses write-to-temp-then-rename pattern to ensure atomic updates
    and prevent data corruption from interrupted writes.
    """
    temp_file = file_path.with_suffix(".tmp")

    try:
        # Write to temporary file
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk

        # Atomic rename (POSIX guarantees atomicity)
        temp_file.replace(file_path)
    except Exception:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""

    name: str
    model_name: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    quantization: str
    dataset_size: int
    notes: str = ""


@dataclass
class ExperimentResults:
    """Results from a training experiment."""

    final_loss: float
    final_eval_loss: Optional[float]
    best_loss: float
    best_eval_loss: Optional[float]
    training_time: float
    total_steps: int
    loss_history: List[float]
    eval_loss_history: List[float]


class ExperimentTracker:
    """
    Track and manage training experiments.

    Stores experiment configurations, metrics, and results to enable
    comparison and analysis across different training runs.

    Size Limits:
    - Max 1000 experiments total
    - Max 10,000 metric entries per experiment
    - Auto-prune oldest experiments when limit reached
    """

    MAX_EXPERIMENTS = 1000
    MAX_METRICS_PER_EXPERIMENT = 10_000

    def __init__(self, experiments_dir: str = "./experiments"):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.experiments_dir / "experiments.json"
        self.current_experiment = None
        self.start_time = None

        # Load existing experiments
        self.experiments = self._load_experiments()

    def _load_experiments(self) -> List[Dict[str, Any]]:
        """Load experiments from disk."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                console.print("[yellow]⚠ Could not load experiments file, starting fresh[/yellow]")
                return []
        return []

    def _save_experiments(self):
        """Save experiments to disk using atomic write."""
        atomic_write_json(self.experiments_file, self.experiments)

    def start_experiment(self, name: str, config: Dict[str, Any], notes: str = "") -> str:
        """
        Start tracking a new experiment.

        Args:
            name: Unique name for the experiment
            config: Experiment configuration dictionary
            notes: Optional notes about the experiment

        Returns:
            Experiment ID
        """
        # Generate unique ID
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_experiment = {
            "id": exp_id,
            "name": name,
            "config": config,
            "notes": notes,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "running",
            "metrics": {
                "loss_history": [],
                "eval_loss_history": [],
                "learning_rate_history": [],
            },
            "results": {},
        }

        self.start_time = time.time()

        console.print(f"\n[bold cyan]Started experiment:[/bold cyan] {exp_id}\n")
        return exp_id

    def log_metrics(
        self, step: int, loss: float, learning_rate: float, eval_loss: Optional[float] = None
    ):
        """
        Log metrics for current step.

        Args:
            step: Current training step
            loss: Training loss
            learning_rate: Current learning rate
            eval_loss: Optional evaluation loss
        """
        if not self.current_experiment:
            console.print("[yellow]⚠ No active experiment[/yellow]")
            return

        metrics = self.current_experiment["metrics"]

        # Check metric count limit
        total_metrics = (
            len(metrics["loss_history"])
            + len(metrics["learning_rate_history"])
            + len(metrics["eval_loss_history"])
        )

        if total_metrics >= self.MAX_METRICS_PER_EXPERIMENT:
            console.print(
                f"[yellow]⚠ Experiment reached max metrics ({self.MAX_METRICS_PER_EXPERIMENT}), "
                "stopping metric collection[/yellow]"
            )
            return

        metrics["loss_history"].append({"step": step, "value": loss})
        metrics["learning_rate_history"].append({"step": step, "value": learning_rate})

        if eval_loss is not None:
            metrics["eval_loss_history"].append({"step": step, "value": eval_loss})

    def end_experiment(
        self, final_loss: float, final_eval_loss: Optional[float] = None, status: str = "completed"
    ):
        """
        End current experiment and save results.

        Args:
            final_loss: Final training loss
            final_eval_loss: Final evaluation loss
            status: Experiment status (completed, failed, stopped)
        """
        if not self.current_experiment:
            console.print("[yellow]⚠ No active experiment to end[/yellow]")
            return

        training_time = time.time() - self.start_time if self.start_time else 0

        # Calculate results
        loss_values = [m["value"] for m in self.current_experiment["metrics"]["loss_history"]]
        eval_loss_values = [
            m["value"] for m in self.current_experiment["metrics"]["eval_loss_history"]
        ]

        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = status
        self.current_experiment["results"] = {
            "final_loss": final_loss,
            "final_eval_loss": final_eval_loss,
            "best_loss": min(loss_values) if loss_values else final_loss,
            "best_eval_loss": min(eval_loss_values) if eval_loss_values else final_eval_loss,
            "training_time": training_time,
            "total_steps": len(loss_values),
        }

        # Save experiment
        self.experiments.append(self.current_experiment)

        # Auto-prune if we exceed max experiments
        if len(self.experiments) > self.MAX_EXPERIMENTS:
            # Sort by start time and remove oldest
            self.experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)
            removed = self.experiments[self.MAX_EXPERIMENTS :]
            self.experiments = self.experiments[: self.MAX_EXPERIMENTS]
            console.print(
                f"[dim]Pruned {len(removed)} old experiments (max: {self.MAX_EXPERIMENTS})[/dim]"
            )

        self._save_experiments()

        console.print(
            f"\n[bold green]✓ Experiment completed:[/bold green] {self.current_experiment['id']}"
        )
        console.print(f"  Final loss: {final_loss:.4f}")
        console.print(f"  Training time: {training_time:.1f}s\n")

        self.current_experiment = None
        self.start_time = None

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.

        Args:
            exp_id: Experiment ID

        Returns:
            Experiment dictionary or None
        """
        for exp in self.experiments:
            if exp["id"] == exp_id:
                return exp
        return None

    def list_experiments(
        self, status: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments.

        Args:
            status: Filter by status (running, completed, failed)
            limit: Maximum number of experiments to return

        Returns:
            List of experiments
        """
        experiments = self.experiments

        # Filter by status
        if status:
            experiments = [e for e in experiments if e["status"] == status]

        # Sort by start time (most recent first)
        experiments = sorted(experiments, key=lambda x: x["start_time"], reverse=True)

        # Apply limit
        if limit:
            experiments = experiments[:limit]

        return experiments

    def compare_experiments(self, exp_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Compare multiple experiments.

        Args:
            exp_ids: List of experiment IDs to compare

        Returns:
            List of experiment summaries
        """
        summaries = []

        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if not exp:
                console.print(f"[yellow]⚠ Experiment not found: {exp_id}[/yellow]")
                continue

            config = exp["config"]
            results = exp["results"]

            summary = {
                "id": exp["id"],
                "name": exp["name"],
                "model": config.get("model_name", "N/A"),
                "lora_rank": config.get("lora_rank", "N/A"),
                "learning_rate": config.get("learning_rate", "N/A"),
                "final_loss": results.get("final_loss", "N/A"),
                "best_loss": results.get("best_loss", "N/A"),
                "training_time": results.get("training_time", "N/A"),
                "status": exp["status"],
            }

            summaries.append(summary)

        return summaries

    def get_best_experiment(self, metric: str = "final_loss") -> Optional[Dict[str, Any]]:
        """
        Get best experiment by metric.

        Args:
            metric: Metric to compare (final_loss, best_loss, training_time)

        Returns:
            Best experiment or None
        """
        completed = [e for e in self.experiments if e["status"] == "completed"]

        if not completed:
            return None

        # Sort by metric (lower is better for loss metrics)
        if metric in ["final_loss", "best_loss"]:
            best = min(completed, key=lambda x: x["results"].get(metric, float("inf")))
        else:
            best = min(completed, key=lambda x: x["results"].get(metric, float("inf")))

        return best

    def delete_experiment(self, exp_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        for i, exp in enumerate(self.experiments):
            if exp["id"] == exp_id:
                self.experiments.pop(i)
                self._save_experiments()
                console.print(f"[green]✓ Deleted experiment: {exp_id}[/green]")
                return True

        console.print(f"[yellow]⚠ Experiment not found: {exp_id}[/yellow]")
        return False

    def export_experiments(self, output_file: str, exp_ids: Optional[List[str]] = None):
        """
        Export experiments to JSON file.

        Args:
            output_file: Output file path
            exp_ids: Optional list of experiment IDs to export (exports all if None)
        """
        if exp_ids:
            experiments = [self.get_experiment(exp_id) for exp_id in exp_ids]
            experiments = [e for e in experiments if e is not None]
        else:
            experiments = self.experiments

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(experiments, f, indent=2)

        console.print(f"[green]✓ Exported {len(experiments)} experiments to {output_file}[/green]")
