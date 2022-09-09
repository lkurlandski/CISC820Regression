# CISC820Regression
Viet and Luke

CISC-820: Quantitative Foundations

Project 1: Linear Feature Engineering

## Usage
Using Python 3.10.4 and pip, create a virtual environment and install the dependencies

```python -m venv env```

```source env/bin/activate```

```pip install black numpy pytest scikit-learn```

Alternatively, using Python 3.10.4 and conda, create a virtual environment and install the dependencies

```conda create --name py310 python=3.10.4 ```

```source activate py310```

```conda install black numpy pytest scikit-learn```

Once the proper version of Python and dependencies are installed, run the main program with

```python main.py```

the unit tests with

```pytest main.py```

and the auto-formatter with

```black main.py```

## Contents
 - main.py contains the regression program
 - tests.py contains some of the tests performed in class
 - testinputs.txt, traindata.txt are the files provided by the instructor
 - results.json contains results from every version of regression performed
 - testoutputs.txt contains predictions corresponding to testinputs.txt from the best regression variant
