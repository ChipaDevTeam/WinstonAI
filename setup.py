"""
WinstonAI - GPU-optimized Reinforcement Learning Trading Bot
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='winston-ai',
    version='1.1.0',
    author='ChipaDevTeam',
    author_email='',
    description='GPU-optimized Reinforcement Learning Trading Bot for Binary Options',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChipaDevTeam/WinstonAI',
    project_urls={
        'Bug Reports': 'https://github.com/ChipaDevTeam/WinstonAI/issues',
        'Source': 'https://github.com/ChipaDevTeam/WinstonAI',
        'Documentation': 'https://github.com/ChipaDevTeam/WinstonAI/blob/main/README.md',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
    ],
    keywords='trading, forex, binary-options, reinforcement-learning, deep-learning, pytorch, gpu, cuda, ai, machine-learning',
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'pylint>=2.17.0',
            'mypy>=1.4.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'winston-train=train_gpu_optimized:main',
            'winston-trade=ultra_live_trading_bot:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.json', '*.md'],
    },
    zip_safe=False,
)
