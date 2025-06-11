"""Setup script for sales forecasting package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sales-forecasting",
    version="0.1.0",
    author="Sales Forecasting Team",
    author_email="team@example.com",
    description="Deep learning sales forecasting for Chinese e-commerce platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/sales-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "statsmodels>=0.12.0",
        "openpyxl>=3.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
    ],
    extras_require={
        "deep_learning": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "transformers>=4.10.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "sales-forecast=scripts.train:main",
            "sales-preprocess=scripts.preprocess:main",
        ],
    },
) 