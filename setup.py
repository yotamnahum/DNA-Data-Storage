from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="srrtransfomer",
    version="0.1",
    description="Single-Read Reconstruction for DNA Data Storage Using Transformers",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="To_Complte",
    author_email="To_Complte",
    url="To_Complte",
    #packages=find_packages(exclude="tests"),  # same as name
    license="MIT",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.6",
)