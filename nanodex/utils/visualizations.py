"""
Terminal-based visualizations for training metrics.

Provides Rich-based visualizations for loss curves, learning rates,
and other training metrics that work in the terminal.
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.text import Text
import math

console = Console()


class MetricsVisualizer:
    """Visualize training metrics in the terminal."""

    @staticmethod
    def sparkline(data: List[float], width: int = 20) -> str:
        """
        Create a simple sparkline visualization.

        Args:
            data: List of numeric values
            width: Width of the sparkline in characters

        Returns:
            String representation of sparkline
        """
        if not data or len(data) == 0:
            return ""

        # Sparkline characters from low to high
        chars = "▁▂▃▄▅▆▇█"

        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            return chars[0] * min(len(data), width)

        # Normalize and map to characters
        normalized = []
        for val in data[-width:]:  # Take last `width` points
            norm = (val - min_val) / (max_val - min_val)
            idx = int(norm * (len(chars) - 1))
            normalized.append(chars[idx])

        return "".join(normalized)

    @staticmethod
    def format_number(num: float, precision: int = 4) -> str:
        """Format number for display."""
        if num == 0:
            return "0.0"
        elif abs(num) < 0.01:
            return f"{num:.2e}"
        else:
            return f"{num:.{precision}f}"

    @staticmethod
    def show_training_metrics(
        epoch: int,
        total_epochs: int,
        loss: float,
        learning_rate: float,
        loss_history: List[float],
        eval_loss: Optional[float] = None
    ):
        """
        Display current training metrics.

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            loss: Current training loss
            learning_rate: Current learning rate
            loss_history: History of loss values
            eval_loss: Optional evaluation loss
        """
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green")
        table.add_column("Trend", style="yellow", width=25)

        # Epoch progress
        progress = f"{epoch}/{total_epochs}"
        progress_bar = "█" * int((epoch / total_epochs) * 20) + "░" * (20 - int((epoch / total_epochs) * 20))
        table.add_row("Epoch", progress, progress_bar)

        # Training loss
        loss_sparkline = MetricsVisualizer.sparkline(loss_history) if loss_history else ""
        table.add_row(
            "Training Loss",
            MetricsVisualizer.format_number(loss),
            loss_sparkline
        )

        # Evaluation loss
        if eval_loss is not None:
            table.add_row(
                "Eval Loss",
                MetricsVisualizer.format_number(eval_loss),
                ""
            )

        # Learning rate
        table.add_row(
            "Learning Rate",
            MetricsVisualizer.format_number(learning_rate, precision=6),
            ""
        )

        console.print(table)

    @staticmethod
    def show_loss_comparison(
        losses: Dict[str, List[float]],
        title: str = "Loss Comparison"
    ):
        """
        Compare multiple loss curves.

        Args:
            losses: Dictionary mapping names to loss histories
            title: Title for the comparison
        """
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Initial", justify="right")
        table.add_column("Final", justify="right")
        table.add_column("Change", justify="right", style="green")
        table.add_column("Trend", style="yellow", width=30)

        for name, loss_history in losses.items():
            if not loss_history or len(loss_history) == 0:
                continue

            initial = loss_history[0]
            final = loss_history[-1]
            change = ((final - initial) / initial * 100) if initial != 0 else 0

            change_str = f"{change:+.1f}%"
            if change < 0:
                change_str = f"[green]{change_str}[/green]"
            else:
                change_str = f"[red]{change_str}[/red]"

            sparkline = MetricsVisualizer.sparkline(loss_history, width=30)

            table.add_row(
                name,
                MetricsVisualizer.format_number(initial),
                MetricsVisualizer.format_number(final),
                change_str,
                sparkline
            )

        console.print(table)

    @staticmethod
    def show_experiment_summary(experiments: List[Dict[str, Any]]):
        """
        Show summary of multiple experiments.

        Args:
            experiments: List of experiment dictionaries with metrics
        """
        table = Table(title="Experiment Comparison", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("LoRA Rank", justify="right")
        table.add_column("Learning Rate", justify="right")
        table.add_column("Final Loss", justify="right", style="green")
        table.add_column("Duration", justify="right")

        for exp in experiments:
            name = exp.get('name', 'N/A')
            model = exp.get('model', 'N/A')
            lora_rank = str(exp.get('lora_rank', 'N/A'))
            learning_rate = MetricsVisualizer.format_number(
                exp.get('learning_rate', 0),
                precision=6
            )
            final_loss = MetricsVisualizer.format_number(
                exp.get('final_loss', 0)
            )
            duration = exp.get('duration', 'N/A')

            table.add_row(
                name,
                model,
                lora_rank,
                learning_rate,
                final_loss,
                duration
            )

        console.print(table)

    @staticmethod
    def show_hyperparameter_grid(
        param_name: str,
        param_values: List[Any],
        metrics: List[float],
        metric_name: str = "Loss"
    ):
        """
        Visualize hyperparameter search results.

        Args:
            param_name: Name of hyperparameter
            param_values: List of hyperparameter values tested
            metrics: Corresponding metric values
            metric_name: Name of the metric
        """
        if len(param_values) != len(metrics):
            console.print("[red]Error: param_values and metrics must have same length[/red]")
            return

        table = Table(
            title=f"Hyperparameter Search: {param_name}",
            box=box.ROUNDED
        )
        table.add_column(param_name, style="cyan")
        table.add_column(metric_name, justify="right", style="green")
        table.add_column("Relative", justify="right")

        # Find best
        best_idx = metrics.index(min(metrics))

        for i, (param, metric) in enumerate(zip(param_values, metrics)):
            param_str = str(param)
            metric_str = MetricsVisualizer.format_number(metric)

            if i == best_idx:
                param_str = f"[bold green]★ {param_str}[/bold green]"
                metric_str = f"[bold green]{metric_str}[/bold green]"

            # Relative performance
            relative = (metric - metrics[best_idx]) / metrics[best_idx] * 100
            relative_str = f"+{relative:.1f}%" if relative > 0 else "0.0%"

            table.add_row(param_str, metric_str, relative_str)

        console.print(table)

    @staticmethod
    def show_checkpoint_summary(checkpoints: List[Dict[str, Any]]):
        """
        Show summary of saved checkpoints.

        Args:
            checkpoints: List of checkpoint information
        """
        table = Table(title="Training Checkpoints", box=box.ROUNDED)
        table.add_column("Step", style="cyan", justify="right")
        table.add_column("Loss", justify="right", style="green")
        table.add_column("Eval Loss", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Path", style="dim")

        for ckpt in checkpoints:
            step = str(ckpt.get('step', 'N/A'))
            loss = MetricsVisualizer.format_number(ckpt.get('loss', 0))
            eval_loss = MetricsVisualizer.format_number(
                ckpt.get('eval_loss', 0)
            ) if ckpt.get('eval_loss') else "N/A"
            size = ckpt.get('size', 'N/A')
            path = ckpt.get('path', 'N/A')

            table.add_row(step, loss, eval_loss, size, path)

        console.print(table)


class TrainingMonitor:
    """Real-time training monitor with live updates."""

    def __init__(self):
        self.loss_history = []
        self.eval_loss_history = []
        self.lr_history = []
        self.current_epoch = 0
        self.total_epochs = 0

    def update(
        self,
        epoch: int,
        loss: float,
        learning_rate: float,
        eval_loss: Optional[float] = None
    ):
        """Update metrics."""
        self.current_epoch = epoch
        self.loss_history.append(loss)
        self.lr_history.append(learning_rate)

        if eval_loss is not None:
            self.eval_loss_history.append(eval_loss)

    def display(self):
        """Display current metrics."""
        MetricsVisualizer.show_training_metrics(
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            loss=self.loss_history[-1] if self.loss_history else 0,
            learning_rate=self.lr_history[-1] if self.lr_history else 0,
            loss_history=self.loss_history,
            eval_loss=self.eval_loss_history[-1] if self.eval_loss_history else None
        )

    def summary(self):
        """Display training summary."""
        if not self.loss_history:
            console.print("[yellow]No training data to display[/yellow]")
            return

        losses = {
            "Training Loss": self.loss_history
        }

        if self.eval_loss_history:
            losses["Evaluation Loss"] = self.eval_loss_history

        MetricsVisualizer.show_loss_comparison(losses, title="Training Summary")
