#! /usr/bin/python
from setuptools import setup
from io import open
# to install type:
# python setup.py install --root=/


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()


setup(name='ltck', version='0.1.0',
      author='Mohamed Mahrous',
      author_email='m.mahrous.94@gmail.com',
      url='https://github.com/lingorithm/lim',
      license='MIT',
      description="Lingorithm Text Classification Kit",
      long_description=readme(),
      long_description_content_type='text/markdown',
      package_dir={'ltck': 'ltck', },
      packages=['ltck'],
      install_requires=[
          'llck'
      ],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Natural Language :: Arabic',
          'Intended Audience :: Developers',
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          'Topic :: Text Processing :: Linguistic',
      ],
      )