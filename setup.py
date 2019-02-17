import sys
from codecs import open
from setuptools import setup

if not sys.version_info[0] == 3:
    sys.exit("\n#####################################\n"
             "  brainsprite does not support python 2.\n"
             "  Please install using python 3.x\n"
             "#####################################\n")

with open('README.md', encoding='utf-8') as fd:
    LONG_DESCRIPTION = fd.read()

setup(
    name='brainsprite.py',
    long_description=LONG_DESCRIPTION,
    license='MIT',
    packages=['.'],
    url='https://github.com/simexp/brainsprite.js',
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'nibabel',
        'nilearn',
        'scipy',
        'sklearn',
        'scikit-image'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.5',
    ],
)
