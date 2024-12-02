from distutils.core import setup


setup(
      name = 'sweeper',
      version = '0.0.1',
      author = 'Alessandro Miano',
      author_email = 'superconducting.nina@gmail.com',
      description = ('N-dimensional sweeper'),
      license = 'GNU General Public License, version 2',
      packages = ['sweeper', 'sweeper.classes','sweeper.colormaps','sweeper.data','sweeper.functions'],
      package_data = {},
      include_package_data = True,
      install_requires = []
      )