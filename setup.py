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
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for estimating and analyzing joint types from point cloud data",
    keywords="computer vision, robotics, point cloud, joint estimation",
    python_requires=">=3.7",
)