# MFM Winter Workshop: Project 1

# Quick Start
You can check the notebook folder for some intial tests and their outputs.
## Unix (Linux/MacOS)
```shell
git clone git@github.com/dominiccook/reto && cd reto
```
```shell
python src/Modern_Portfolio_Success_Rate_MFM_Workshop.py
```

## Windows
#### Command Prompt/CMD
```cmd
git clone git@github.com:dominiccook/reto.git & cd reto
```
```cmd
python src\Modern_Portfolio_Success_Rate_MFM_Workshop.py
```
#### Powershell
```powershell
git clone git@github.com:dominiccook/reto.git; cd reto
```
```powershell
python src\ Modern_Portfolio_Success_Rate_MFM_Workshop.py
```

# Installation
Run the following to install package:
```shell
pip install project1
```
## Available Imports 
It will be helpful to have the package for importing the methods into a jupyter notebook, since jupyter notebooks provide lots of functionality that it helpful when exploring the code, and testing different parameters. 
(NOTE: More parametrization is needed to make the package more portable, and improve its usability in a jupyter notebook.)
*(NOTE: These _will_ change, but I will try to keep the README up-to-date.)*
```python
# main.py consists of the functions used
#+to wrangle the return data and calculate
#+the metrics entered into the other modules.
import main

# The following module contains the scripts
#+that do the actual simulating.
# Uses metrics calculated from calculated
#+databento returns by default.
import Modern_Portfolio_Success_Rate as mpsr
```

# Usage
### Parameters and Data Processing
Running the main source file will print logs that show the data used from databento, and the calculations done to find the parameters put into the mentor script.

The return data is NOT included in this repository. If you would like to run this with actual return data, the main.py file assumes OHLCV schema for hourly returns. However, the necessary return metrics and correlations are included as CSV files in this repository and can be found in the data subdirectory.

```shell
python src/main.py
```

![Index Fund Return Calculations from Hourly Return Data](examples/main_output.png "Terminal Output")

### Simulation
Running this Portfolio Success Rate script will simulate future portfolio successes, indicate the portfolio with the highest success rate across 1000 paths, and print the optimal weights associated with that most successful portfolio.

```shell
python src/Modern_Portfolio_Success_Rate_MFM_Workshop.py
``` 
 ![Student Simulation](examples/figure_1_databento.png "Portfolio Success Rate")