from setuptools import setup, find_packages

setup(
    name="sentinel_colab",                # Package name
    version="0.1.0",                   # Version
    author="Pedro Silva",                # Replace with your name
    author_email="pedrosilva7320@gmail.com",    # Replace with your email
    description="A Colab-friendly package to speed up training of RF-DETR and YOLOv11 models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/phsilvarepo/sentinel-colab",  # Replace with your repo URL
    packages=find_packages(),          # Automatically find all packages
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",                  # Minimal requirement, you can add torchvision if needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,         # Include README and other files
)
