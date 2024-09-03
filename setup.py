from setuptools import find_packages, setup

packages = filter(lambda x: x.startswith("neural_irt"), find_packages())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-irt",
    version="0.1",
    description="A PyTorch based NeuralNet framework to learn and evaluate IRT-based models.",
    author="Maharshi Gor",
    author_email="mgor@cs.umd.edu",
    url="https://github.com/maharshi95/neural-irt",
    project_urls={
        "Bug Tracker": "https://github.com/maharshi95/neural-irt/issues",
    },
    # Choose a limiting License that doesn't allow making business product
    license="GPLv2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=packages,
    install_requires=[
        "pytest",
        "torch>=2.0",
        "pytorch-lightning",
        "datasets",
        "rich",
        "rich-argparse",
        "loguru",
        "msgspec",
        "pydantic>=2.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3.10",
    ],
)
