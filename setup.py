from setuptools import setup, find_packages

setup(
    name = "Measuring_Statistical_Heterogeneity",
    version="0.1.1",
    authors = [],
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=[

    ],
    entry_points={
        'console_scripts': [
            "run_statistical = src.statistical_divergence.statistical_divergence:run_experiment",
            "run_multi_statistical = src.statistical_divergence.statistical_divergence:multi_plot",
            "run_ml = src.ml.ml_mnist_cifar10:run_experiment",
            "run_ml_multi = src.ml.ml_mnist_cifar10:run_multi_experiment"
        ]
    }
)