from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='C2F-Seg as a package',
    name='c2f_seg',
    packages=find_packages(),
    install_requires=[
        'numpy==1.22.4',
        "opencv-python",
        "tqdm",
        "matplotlib",
        "pandas",
        "packaging",
        "tensorboard",
        "cvbase",
        "scikit-image",
        "pycocotools",
    ],

)