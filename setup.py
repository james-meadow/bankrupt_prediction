import logging
import subprocess
import setuptools
from setuptools.command.install import install
from setuptools import find_packages
from setuptools import setup

class CustomCommands(install):
  """A setuptools Command class able to run arbitrary commands."""

  def RunCustomCommand(self, command_list):
    print ('Running command: %s' % command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print ('Command output: %s' % stdout_data)
    logging.info('Log command output: %s', stdout_data)
    if p.returncode != 0:
      raise RuntimeError('Command %s failed: exit code: %s' %
                         (command_list, p.returncode))

  def run(self):
    self.RunCustomCommand(
    ['pip2.7', 'install','google-cloud-storage'])
    install.run(self)


REQUIRED_PACKAGES = [
	'pandas',
	'tensorflow',
	'numpy',
    'google-cloud-storage'
]

setup(
    name='trainer',
    version='0.1',
 #   install_requires='data_set_helpers.py',
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.',
    cmdclass={
         # Command class instantiated and run during install scenarios.
        'install': CustomCommands,
    }
)
