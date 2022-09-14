# CISC820Regression
Viet and Luke

CISC-820: Quantitative Foundations

Project 1: Linear Feature Engineering

## Environment Setup
The easiest way to install the dependencies is to use conda,

```conda init```

```conda create --name py310 python=3.10.4 ```

```conda activate py310```

```conda install black numpy pytest scikit-learn```

```conda install -c pytorch pytorch```

Alternatively, using Python 3.10.4, with pip and env,

```python -m venv env```

```source env/bin/activate```

```pip install black numpy pytest scikit-learn torch==1.12.1```

## Running the Program
For the TA grading the assignment, this is likely the only command you will need to run after setting up the environment. To reproduce the results produced for submission, run

```python main.py --submission```
the reuslt will be in the file `testoutputs_nn.txt`.

To run a bulk set of experiments, run

```python main.py```

The settings of the experiments can be tweaked with command line arguments, such as

```python main.py -k 10 -p 3 -r results.json -s -v```

A brief description of the flags can be found by running

```python main.py --help```

For developers, the unit tests are run with

```pytest main.py```

and the auto-formatter with

```black *.py```
