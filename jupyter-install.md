
## <a name="PythonJupyter">Python and Jupyter</a>

This is the easiest, 

see: https://www.anaconda.com/products/individual

This is the "individual edition", which is the Free, Open Source edition. 

It includes Python 3-point-something. Python version 2 and version 3 are slightly
incompatible. Choose 3 if ever you have a choice.

The anaconda also includes,

1. Python3
1. Jupyter, the notebook/browser interface to the scientific computing world
1. Packages. Lots of packages, such as Numby, Pandas, Matplotlib
1. The conda package manager
1. The conda python virtual environment system

This should run on everything. Mac, Windows and Linux. That is nice.

#### Using Jupyter

Once git has been installed (see below), bring the REU-CFS githup repo to your local machine, 

> `mkdir reu-cfs ; cd to reu-cfs ; git clone https://github.com/burtr/reu-cfs.git`

Then start the notebook,

> `jupyter notebook`

This works on the terminal of MacOS, and from the Anaconda Prompt on windows.

If auto_actitvates as set false:

> `conda config --set auto_activate_False`

then activate it now,

> `conda activate`

If conda can not be found, then use the full pathname, or adjust your path environment variable. 
The activate command is often found at `~/anaconda3/bin/activate`, where the tilde is 
Unix-speak for "my home directory".

For more information, see

> https://docs.anaconda.com/anaconda/install/linux/
