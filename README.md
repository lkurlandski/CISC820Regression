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


If you want to test the neural network, install these dependencies:
For Python and pip:
```
# For Window and Mac:
pip install torch torchvision torchaudio
# For Linux
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

For Anaconda:
```
# For Window and Linux:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# For Mac:
conda install pytorch torchvision torchaudio -c pytorch
```

## Running the normal linear regression
Once the proper version of Python and dependencies are installed, run the main program with

```python main.py```

Query the options for the main program with

```python main.py --help```

the unit tests with

```pytest main.py```

and the auto-formatter with

```black main.py```

## Running with neural network
If you want to train the models for 10 folds,
```
python neural_net_train.py
```
This will automatically export `history.npz` where there is the mse losses and best model data. It also exports 10 models for 10 folds in the `./models` folder. After that, run:

```
python neural_net_eval.py
```
This will pick the best model to export the results.

## Contents
 - linear_regression.py: contains the linear regression model
 - main.py: drives the experiments
 - neural_net.py: contains the neural network model (wrapper)
 - results.json: results from every version of regression performed
 - testinputs.txt: file provided by the instructor
 - testoutputs.txt: predictions corresponding to testinputs.txt from the best regression variant
 - tests.py contains some of the tests performed in class
 - traindata.txt: file provided by the instructor
 - transforms.py: feature engineering functions
