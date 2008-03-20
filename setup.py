from setuptools import setup, find_packages

kws=dict(
    )

setup(name='motmot.ufmf',
      version='0.1',
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      namespace_packages = ['motmot'],
      packages = find_packages(),
      )
