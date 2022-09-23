from setuptools import find_packages, setup

setup(
    name = "graphoflife",
    version = "1.0.0",
    description = "Using LabGraph to simulate John Conway's Game of Life",
    packages = find_packages(),
    python_requires = ">=3.6",
    install_requires = [
        "labgraph==2.0.0",
        "python-opencv>=4.6.0"
    ],
    include_package_data = True
)