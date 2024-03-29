import os
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

# Install requirements
install_requires = []


setup_requirements = ["pytest-runner", "setuptools_scm"]

# Test requirements
test_requirements = [
    "pytest",
    "pandas",
    "numpy",
    "scikit-learn"
]

setup(
    name="pyrust-decision-tree",
    packages=find_packages(),
    author="Miles Granger",
    author_email="miles59923@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Decision tree implemented in Rust w/ Python bindings",
    install_requires=install_requires,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite="tests",
    use_scm_version={
        "write_to": os.path.join("pyrus_decision_tree", "_version.py"),
        "relative_to": __file__,
    },
    rust_extensions=[
        RustExtension(
            "pyrus_decision_tree.pyrus_decision_tree",
            binding=Binding.PyO3,
            path=os.path.join(
                os.path.dirname(__file__), "Cargo.toml"
            ),
        )
    ],
    zip_safe=False,
)