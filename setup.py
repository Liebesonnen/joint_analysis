from setuptools import setup, find_packages

setup(
    name="joint_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "polyscope>=1.3.0",
        "dearpygui>=1.8.0",
        "matplotlib>=3.5.0",
    ],
    author="Rui Cheng",
    author_email="cr1744733299@gmail.com",
    description="A package for estimating and analyzing joint types from dense point trajectories",
    keywords="computer vision, robotics, joint estimation",
    python_requires=">=3.7",
)