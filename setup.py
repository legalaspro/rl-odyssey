from setuptools import setup, find_packages

setup(
    name="rl_odyssey",
    version="0.1",
    description="A reinforcement learning experiments package",
    author="Dmitri Manajev",
    packages=find_packages(),
    install_requires=[
        "gymnasium==1.0.0",
        "gymnasium[atari]",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "imageio",
        "tensorboard",
        "dm-control",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)