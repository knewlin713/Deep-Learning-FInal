# Deep-Learning-FInal

Setting up the environment:
Run through requriments.txt. This most likely will not work due to the different versions of the python, however for the model to work we need to change the computers version of python. To remedy this problem run requirements 2 and 3 seperately. Which will prompt conda to install a older version of python by using pip and conda seperately because the most likley case that the code is not running is because in requirements everything was installing but not running or it installed half way and then stopped running. Conda will run normally and install what it needs to while pip will install every thing that conda could not install and run.

Setting up system PATHS: 
We understand that even with both requirments 2 and 3 running and properly installed the code might still not run. This might be due to a system paths issue and will require some extra set and fixing. Our model only runs on one computer and struggles to run on other groupmates computer because of the system differences. 

Here are some tips to try to get the model to run

Windows: 
-Run the Installer: Once the download is complete, run the installer. Make sure to check the box that says "Add Python X.X to PATH"
-open the Command Prompt and type python --version to verify that Python has been installed correctly
-Set up a Virtual Environment: It's good practice to work within a virtual environment, especially for larger projects or when working with multiple Python projects. In requirements 2 and 3 we named this environment deepfried. So run conda activate deepfried

**If the steps above do not work, the computer might need to be manually switch the the older outdated version of python. Navigate to settings PATHS/ENVIRONMENTS and try move the old python path to the top of the list.

** this cause alot of other processes to crash in one of our other computers so this is not reccommended

MAC:
-we reccommend the installation of home brew to help with installing the older version of python
