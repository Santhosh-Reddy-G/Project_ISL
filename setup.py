from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="humanoid_locomotion",
    version="0.1.0",
    author="IIT Tirupati - Intelligent Systems Lab",
    description="Deep RL for Humanoid Locomotion from Image-Defined Initial Pose",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)