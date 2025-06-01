# setup.py: Tells Python how to install your project as a package

# find_packages(): Scans for any sub-packages (folders with __init__.py) and includes them

# __init__.py: Declares “this is a package” so that Python includes it during import and installation
'''
setup.py is a special file used to package and distribute a Python project.
It tells Python (and tools like pip) how to install your code as a module or library.

🧱 Why is it Needed?
Makes your code installable using pip install .

Allows others (or servers) to install your project with all dependencies

Makes your code reusable across projects (like a mini pandas or numpy)

Helps structure ML/data science projects like real Python packages

'''
from setuptools import find_packages, setup 
from typing import List
HYPHEN_E_DOT='-e .'
# file_path: str means the parameter file_path must be a string (like "requirements.txt").-> List[str] means the function will return a list of strings.

def get_requirements(file_path:str)->List[str]:
    '''
    this funciton will return the list of requirements
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
#we dont need the e . to be returned while we call this get_requirements function to the install_requires. we only need e . so that 

setup(
name = 'mlproject',
version = '0.0.1',
author = 'suraj',
author_email = 'suraj2005jan@gmail.com',
packages = find_packages(),
# this find packages is used to see , in the src folder, how many sub directories have the __init__.py and it recognises them as a Package and we can do the from this this import this this thanks to find packages and init,py. it recognises src as a importable package.
install_requires = get_requirements('requirements.txt')
#instead of altering it here everytime make a function get_requirements
)