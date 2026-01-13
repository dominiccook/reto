# MFM Winter Workshop: Project 1
## Installation

```shell
pip install -e .
```

## Running
### Parameters and Data Processing
Running the main source file will print logs that show the data I've used from databento, and the calculations I'm doing to find the parameters I'm putting in our mentor's script.
```shell
python src/main.py
```

![Index Fund Return Calculations from Hourly Return Data](examples/main_output.png "Terminal Output")

### Simulation
Running these Portfolio Success Rate scripts will simulate future portfolio successes, either with the Original parameters provided by our mentor, or with the parameters calculated by our script.

```shell
python src/mentor/Modern_Portfolio_Success_Rate_MFM_Workshop.py
```

 ![Mentor Simulation](examples/Figure_1.png "Portfolio Success Rate | Mentor")


```shell
python src/databento/Modern_Portfolio_Success_Rate_MFM_Workshop.py
``` 
 ![Student Simulation](examples/figure_1_databento.png "Portfolio Success Rate | Mentor")



## TODO
- Increase portability of the code.
- Come up with ways of improving our model.
