from setuptools import setup,find_packages 
import pathlib 
import pkg_resources

PKG_NAME="Inprocessing_debiasing"
VERSION="0.01"
EXTRAS = {} 

def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras

setup(
    name=PKG_NAME,
    version=VERSION,
    author=" ",
    url=None,
    description="research project",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning","Debiasing"],
    license="TBD",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 0  Debug mode",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3.`0",
    ],
)
