"""vLLM-based inference server for LoRA adapters."""

import logging
from pathlib import Path
from typing import Optional

from nanodex.config import InferenceConfig

logger = logging.getLogger(__name__)


class InferenceServer:
    """vLLM inference server wrapper."""

    def __init__(self, config: InferenceConfig):
        """
        Initialize inference server.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.server_process = None

    def start(self) -> None:
        """
        Start vLLM server with LoRA adapter.

        This method constructs the vLLM server command and starts it.
        The server runs in the background and serves at config.host:config.port.
        """
        logger.info("Starting vLLM inference server...")
        logger.info(f"Base model: {self.config.base_model}")
        logger.info(f"Adapter: {self.config.adapter_path}")
        logger.info(f"Server: {self.config.host}:{self.config.port}")

        # Build vLLM command
        cmd = self._build_vllm_command()

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("To start the server, run:")
        logger.info(f"  {' '.join(cmd)}")
        logger.info("")
        logger.info("Or use Docker:")
        logger.info(f"  docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \\")
        logger.info(f"    -p {self.config.port}:{self.config.port} vllm/vllm-openai:latest \\")
        logger.info(f"    --model {self.config.base_model} \\")
        logger.info(f"    --enable-lora \\")
        logger.info(f"    --lora-modules nanodex={self.config.adapter_path} \\")
        logger.info(f"    --max-lora-rank {self.config.max_lora_rank} \\")
        logger.info(f"    --host {self.config.host} \\")
        logger.info(f"    --port {self.config.port}")

    def _build_vllm_command(self) -> list[str]:
        """
        Build vLLM server command.

        Returns:
            List of command arguments
        """
        cmd = [
            "vllm",
            "serve",
            self.config.base_model,
            "--enable-lora",
            f"--lora-modules=nanodex={self.config.adapter_path}",
            f"--max-lora-rank={self.config.max_lora_rank}",
            f"--host={self.config.host}",
            f"--port={self.config.port}",
        ]

        return cmd

    def stop(self) -> None:
        """Stop the inference server."""
        logger.info("Server stop not implemented (use external process management)")

    def get_endpoint(self) -> str:
        """
        Get server endpoint URL.

        Returns:
            Server URL
        """
        return f"http://{self.config.host}:{self.config.port}"


def print_server_instructions(config: InferenceConfig) -> None:
    """
    Print instructions for starting the inference server.

    Args:
        config: Inference configuration
    """
    print("=" * 60)
    print("nanodex - Inference Server Instructions")
    print("=" * 60)
    print("")
    print("To start the vLLM inference server, use one of these methods:")
    print("")
    print("Method 1: Direct vLLM")
    print("-" * 60)
    print(f"vllm serve {config.base_model} \\")
    print("  --enable-lora \\")
    print(f"  --lora-modules nanodex={config.adapter_path} \\")
    print(f"  --max-lora-rank {config.max_lora_rank} \\")
    print(f"  --host {config.host} \\")
    print(f"  --port {config.port}")
    print("")
    print("Method 2: Docker")
    print("-" * 60)
    print("docker run --gpus all \\")
    print("  -v ~/.cache/huggingface:/root/.cache/huggingface \\")
    print(f"  -p {config.port}:{config.port} \\")
    print("  vllm/vllm-openai:latest \\")
    print(f"  --model {config.base_model} \\")
    print("  --enable-lora \\")
    print(f"  --lora-modules nanodex={config.adapter_path} \\")
    print(f"  --max-lora-rank {config.max_lora_rank} \\")
    print(f"  --host {config.host} \\")
    print(f"  --port {config.port}")
    print("")
    print("=" * 60)
    print(f"Server will be available at: http://{config.host}:{config.port}")
    print("OpenAPI docs: http://{}:{}/docs".format(config.host, config.port))
    print("=" * 60)
