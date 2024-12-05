from setuptools import setup, find_packages

setup(
    name="hpc-tusimple",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.5',
        'opencv-python>=4.5.3',
        'matplotlib>=3.4.3',
        'seaborn>=0.11.2',
        'pyyaml>=5.4.1',
        'tqdm>=4.62.3',
        'psutil>=5.8.0',
        'gputil>=1.4.0',
        'pandas>=1.3.3',
        'kagglehub>=0.1.0',
        'pillow>=8.3.2',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-cov>=2.12.1',
            'black>=21.9b0',
            'flake8>=3.9.2',
            'mypy>=0.910',
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="HPC implementation of TuSimple Lane Detection",
    keywords="deep-learning, lane-detection, HPC, parallel-computing",
    python_requires='>=3.7',
)