"""
Setup script for DataCruiser
"""

from setuptools import setup, find_packages

setup(
    name="datacruiser",
    version="1.0.0",
    description="RAG-based Footfall Analysis API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="DataCruiser Team",
    author_email="team@datacruiser.com",
    url="https://github.com/datacruiser/datacruiser-server",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "flask>=3.0.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0,<2.0.0",
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "faiss-cpu>=1.8.0",
        "openai>=1.35.0",
        "pydantic>=2.9.0",
        "requests>=2.32.0",
        "urllib3>=2.2.0",
        "certifi>=2024.8.0",
        "charset-normalizer>=3.3.0",
        "idna>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "black>=24.4.0",
            "flake8>=7.1.0",
            "mypy>=1.10.0",
            "pytest-cov>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "datacruiser=datacruiser.app_factory:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
