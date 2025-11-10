"""Setup script for nanodex."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nanodex",
    version="0.2.0",
    author="NoodleMind",
    description="Fine-tune open source coding models on your codebase with RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noodlemind/nanodex",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "ollama>=0.1.0",
        "huggingface-hub>=0.19.0",
        "gitpython>=3.1.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
    ],
    entry_points={
        "console_scripts": [
            "nanodex=nanodex.__main__:main",
        ],
    },
)
