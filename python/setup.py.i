import sys
from setuptools import setup, find_packages

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False


except ImportError:
    bdist_wheel = None

if sys.version_info < (3, 0):
    sys.exit("Sorry, Python < 3.0 is not supported")

setup(
    name="otter",
    version="${PACKAGE_VERSION}",
    author="duncan",
    author_email="duncan11514@gmail.com",
    maintainer="duncan",
    maintainer_email="duncan11514@gmail.com",
    description="otter is a neural network inference framework for learning.",
    url="https://github.com/chenjunhao0315/Tensor",
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    license="BSD-3",
    python_requires=">=3.5",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={"otter": ["otter${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}"]},
    cmdclass={"bdist_wheel": bdist_wheel},
)
