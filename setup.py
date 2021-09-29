from setuptools import find_packages
from setuptools import setup

setup(
    name="estimagic",
    version="0.1.4",
    description="Tools for the estimation of (structural) econometric models.",
    long_description="""
        Estimagic is a Python package that helps to build high-quality and user
        friendly implementations of (structural) econometric models.

        It is designed with large structural models in mind. However, it is also
        useful for any other estimator that numerically minimizes or maximizes a
        criterion function (Extremum Estimator). Examples are maximum likelihood
        estimation, generalized method of moments, method of simulated moments and
        indirect inference.""",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "econometrics",
        "statistics",
        "estimation",
        "extremum estimation",
        "optimization",
        "inference",
        "numerical differentiation",
        "finite differences",
        "richardson extrapolation",
        "derivative free optimization",
        "method of simulated moments",
        "maximum likelihood",
    ],
    url="https://github.com/OpenSourceEconomics/estimagic",
    author="Janos Gabler",
    author_email="janos.gabler@gmail.com",
    packages=find_packages(where="src"),
    entry_points={"console_scripts": ["estimagic=estimagic.cli:cli"]},
    zip_safe=False,
    package_data={"estimagic": ["optimization/algo_dict.json"]},
    include_package_data=True,
    package_dir={"": "src"},
)
