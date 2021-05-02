from setuptools import setup

requirements = [
    "pandas",
    "matplotlib",
    "seaborn",
    "tqdm",
    "imagehash",
    "scikit-image",
    "Pillow",
    "opencv-python",
    "torch",
    "torchvision",
    "pytorch-lightning"
]

setup(
    name='shopee',
    version='1.0',
    packages=[''],
    url='',
    license='',
    author='merrin',
    author_email='',
    description='Shopee-Kaggle',
    install_requires=requirements
)
