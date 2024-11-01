import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements_dev.txt") as f:
    requirements_dev = f.read().splitlines()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="threed_utils",
    version="0.0.1",
    author="Iurilli lab",
    packages=setuptools.find_namespace_packages(exclude=("docs", "tests*")),
    install_requires=requirements,
    extras_require=dict(dev=requirements_dev),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    url="https://github.com/iurillilab/3d-setup",
)
