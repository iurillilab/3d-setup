import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="threed_utils",
    version="0.0.1",
    author="Iurilli lab",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tqdm",
        "matplotlib",
    ],
    url="https://github.com/iurillilab/3d-setup",
)
