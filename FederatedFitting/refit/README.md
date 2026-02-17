This code generate an MCMC chain and generates a number of plots, such as the MCMC likelihood as a function of chain step, distributions of the posteriors, and lightcurves for the medians of the posteriors.

To run, follow these steps:

* Create and activate a virtual environment:

    ```bash
    $ python3 -m venv .venv
    $ source .venv/bin/activate
    ```

* Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

* Decide whether to run with FiestaEM or Afterglowpy and edit the
  relevant constant on the first line of `template/run_mcmc.py`

    ```Python
    FIESTA = True
    # or
    FIESTA = False  # This will use afterglowpy
    ```

* Run the sampling (can be done *without* a GPU):

    ```bash
    cd template
    ./run-fitting.sh
    ```

The outputs will be produced in the `out/` subdirectory.

The input file is obtained by combining the two sites in `examples/fitting_dataset/GW70817_data` and commenting out frequencies that Fiesta cannot deal with (< 1GHz).
