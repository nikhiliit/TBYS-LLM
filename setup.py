"""Setup script for Qwen Agent."""

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='qwen-agent',
    version='1.0.0',
    description='A modern web interface for Qwen language models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Qwen Agent Team',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'qwen-agent=src.app:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='qwen llm ai chat interface web',
)
