"""
Solar Panel Degradation Model - Setup Script
===========================================

Installation script for the solar panel degradation modeling tool.
Provides both development and installation modes.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Optional dependencies
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.910'
    ],
    'pdf': [
        'reportlab>=3.6.0',
        'weasyprint>=54.0'
    ],
    'orbital': [
        'sgp4>=2.19',
        'skyfield>=1.39'
    ],
    'full': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'reportlab>=3.6.0',
        'weasyprint>=54.0',
        'sgp4>=2.19',
        'skyfield>=1.39'
    ]
}

setup(
    name="solar-panel-degradation",
    version="1.0.0",
    author="Solar Panel Degradation Team",
    author_email="contact@solarpanelmodel.com",
    description="User-friendly solar panel degradation modeling for space applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ezra-compyle/solar-panel-degradation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
        "config": ["*.json"],
        "frontend": ["**/*.html", "**/*.css", "**/*.js"],
    },
    entry_points={
        "console_scripts": [
            "solar-panel-api=src.api.server:run_server",
            "solar-panel-model=src.main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ezra-compyle/solar-panel-degradation/issues",
        "Source": "https://github.com/ezra-compyle/solar-panel-degradation",
        "Documentation": "https://solarpaneldegradation.readthedocs.io/",
    },
    keywords="solar panel degradation space radiation thermal orbit satellite",
    zip_safe=False,
)