from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
setup(
    name="deepclustering2",
    version="2.0.0",
    packages=find_packages(exclude=[".data", "script", "test", "runs", "config"]),
    url="https://github.com/jizongFox/deep-clustering-toolbox",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jizong Peng",
    author_email="jizong.peng.1@etsmtl.net",
    install_requires=[
        "msgpack",
        "numpy",
        "torch",
        "torchvision",
        "Pillow",
        "scikit-learn",
        "behave",
        "requests",
        "scikit-image",
        "pandas",
        "easydict",
        "matplotlib",
        "tqdm",
        "py",
        "tensorboardX",
        "tensorboard",
        "opencv-python",
        "medpy",
        "pyyaml",
        "termcolor",
        "gpuqueue",
        "gdown",
        "torch_optimizer",
    ],
    entry_points={
        "console_scripts": [
            "viewer=deepclustering2.viewer.Viewer:main",
            "clip_screencapture=deepclustering2.postprocessing.clip_images:call_from_cmd",
            "report=deepclustering2.postprocessing.report2:call_from_cmd",
            "file_extractor=deepclustering2.postprocessing.folder_processing:main",
        ]
    },
)
