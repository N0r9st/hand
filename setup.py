from setuptools import find_packages, setup

setup(
    name='hand-v0',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['gym', 'mujoco_py', 'flax','wandb'])