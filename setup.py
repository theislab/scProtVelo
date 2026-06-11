from setuptools import find_packages, setup

setup(
    name="scprotvelo",
    packages=find_packages(),
    install_requires=[
        "scvi-tools==1.1.5",
        "scvelo==0.3.2",
        "scanpy==1.9.1",
        "anndata==0.10.7",
        "matplotlib==3.6.3",
        "pandas==1.5.3",
        "jupyter==1.1.1",
        "matplotlib-inline==0.1.7"
    ],
)
