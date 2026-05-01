VS Code is still trying to run the notebook with system Python 3.14.3:
/usr/bin/python
But your ipykernel is inside myenv, so you need to select/register that venv kernel.
From your project root:
source myenv/bin/activatepython -m pip install ipykernelpython -m ipykernel install --user --name myenv --display-name "Python (myenv)"