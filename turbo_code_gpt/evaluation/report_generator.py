"""
Generate evaluation reports in various formats.
"""

from typing import Dict, Any, List
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports."""

    @staticmethod
    def generate_markdown_report(
        results: Dict[str, Any],
        output_file: str,
        model_name: str = "Fine-tuned Model",
        dataset_info: Optional[Dict[str, Any]] = None
    ):
        """
        Generate a Markdown evaluation report.

        Args:
            results: Evaluation results dict
            output_file: Output file path
            model_name: Name of the model
            dataset_info: Optional dataset information
        """
        report_lines = []

        # Header
        report_lines.append(f"# Evaluation Report: {model_name}")
        report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Dataset info
        if dataset_info:
            report_lines.append("## Dataset Information\n")
            for key, value in dataset_info.items():
                report_lines.append(f"- **{key}**: {value}")
            report_lines.append("")

        # Metrics
        report_lines.append("## Evaluation Metrics\n")
        report_lines.append("| Metric | Score |")
        report_lines.append("|--------|-------|")

        for metric, score in sorted(results.items()):
            if isinstance(score, float):
                report_lines.append(f"| {metric} | {score:.4f} |")
            else:
                report_lines.append(f"| {metric} | {score} |")

        report_lines.append("")

        # Analysis
        report_lines.append("## Analysis\n")

        # Highlight best/worst metrics
        if results:
            numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}

            if numeric_results:
                best_metric = max(numeric_results, key=numeric_results.get)
                worst_metric = min(numeric_results, key=numeric_results.get)

                report_lines.append(f"- **Best Performance**: {best_metric} ({numeric_results[best_metric]:.4f})")
                report_lines.append(f"- **Needs Improvement**: {worst_metric} ({numeric_results[worst_metric]:.4f})")

        # Write report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Markdown report saved to {output_file}")

    @staticmethod
    def generate_json_report(
        results: Dict[str, Any],
        output_file: str,
        model_name: str = "Fine-tuned Model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Generate a JSON evaluation report.

        Args:
            results: Evaluation results dict
            output_file: Output file path
            model_name: Name of the model
            metadata: Optional metadata
        """
        report = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }

        if metadata:
            report['metadata'] = metadata

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"JSON report saved to {output_file}")

    @staticmethod
    def generate_html_report(
        results: Dict[str, Any],
        output_file: str,
        model_name: str = "Fine-tuned Model"
    ):
        """
        Generate an HTML evaluation report.

        Args:
            results: Evaluation results dict
            output_file: Output file path
            model_name: Name of the model
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report: {model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Evaluation Report: {model_name}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Score</th>
        </tr>
"""

        for metric, score in sorted(results.items()):
            if isinstance(score, float):
                html += f"        <tr><td>{metric}</td><td>{score:.4f}</td></tr>\n"
            else:
                html += f"        <tr><td>{metric}</td><td>{score}</td></tr>\n"

        html += """    </table>
</body>
</html>"""

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"HTML report saved to {output_file}")

    @staticmethod
    def compare_results(
        results_list: List[Dict[str, Any]],
        labels: List[str],
        output_file: str
    ):
        """
        Generate a comparison report for multiple evaluation results.

        Args:
            results_list: List of evaluation results dicts
            labels: List of labels for each result set
            output_file: Output file path
        """
        report_lines = []

        # Header
        report_lines.append("# Model Comparison Report")
        report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Get all metrics
        all_metrics = set()
        for results in results_list:
            all_metrics.update(results.keys())

        # Create comparison table
        report_lines.append("## Metric Comparison\n")
        report_lines.append("| Metric | " + " | ".join(labels) + " |")
        report_lines.append("|--------|" + "|".join(["--------"] * len(labels)) + "|")

        for metric in sorted(all_metrics):
            row = [metric]
            for results in results_list:
                score = results.get(metric, "N/A")
                if isinstance(score, float):
                    row.append(f"{score:.4f}")
                else:
                    row.append(str(score))
            report_lines.append("| " + " | ".join(row) + " |")

        # Write report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Comparison report saved to {output_file}")
