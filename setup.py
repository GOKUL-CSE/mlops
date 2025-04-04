from setuptools import find_packages,setup
from typing import List

HYPER_E = '-e .'
def get_requirements(file_path:str)->List[str]:


    requirements = []
    with open(file_path)  as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n",'') for req in requirements]

        if HYPER_E in requirements:
            requirements.remove(HYPER_E)

    return requirements
setup(
    name="mlproject",
    version='0.0.0.1',
    author="Gokul",
    author_email="gokul39164@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')


)