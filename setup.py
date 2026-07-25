from setuptools import setup, find_packages

setup(
    name="dqn-from-scratch",
    version="1.0.0",
    description="A modular, from-scratch implementation of Deep Q-Networks (DQN).",
    packages=find_packages(include=["dqn", "dqn.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "gymnasium>=0.29",
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "matplotlib>=3.7"],
    },
)
