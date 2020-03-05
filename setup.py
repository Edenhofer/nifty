# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from setuptools import find_packages, setup


def write_version():
    import subprocess
    p = subprocess.Popen(["git", "describe", "--dirty", "--tags", "--always"],
                         stdout=subprocess.PIPE)
    res = p.communicate()[0].strip().decode('utf-8')
    with open("nifty6/git_version.py", "w") as file:
        file.write('gitversion = "{}"\n'.format(res))


write_version()
exec(open('nifty6/version.py').read())

setup(name="nifty6",
      version=__version__,
      author="Theo Steininger, Martin Reinecke",
      author_email="martin@mpa-garching.mpg.de",
      description="Numerical Information Field Theory",
      url="http://www.mpa-garching.mpg.de/ift/nifty/",
      packages=find_packages(include=["nifty6", "nifty6.*"]),
      zip_safe=True,
      license="GPLv3",
      setup_requires=['scipy>=1.4.1'],
      install_requires=['scipy>=1.4.1'],
      python_requires='>=3.5',
      classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"],
      )
