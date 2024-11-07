from setuptools import setup
import setuptools
    
setup(
    name="pcfnet",
    version="0.1.0",
    install_requires=[
        "tqdm",
        "numpy",
        "dask",
        "pandas",
        "h5py",
        "tables",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "tensorboard",
        "pyro-ppl",
        ],
    extras_require={
        "all": [
            "wandb<=0.18.5"
        ],
    },
    packages=setuptools.find_packages(),
    description="PCFNet officail implementation",
    author='Yoshihiro Takeda',
    author_email='y.takeda@astron.s.u-tokyo.ac.jp',
    url='https://github.com/YoshihiroTakeda/PCFNet',
    python_requires='>=3.8',
)
