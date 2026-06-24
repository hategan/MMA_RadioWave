# Multimessenger App - Federated Fitting Module, take two

## Overview
This is a simplified version of the federated fitting MMA module in the main branch. The main purpose of this is to parallelize the MCMC likelihood evaluation in order to fix the performance penalty seen with the first version of this code and plain local emcee (a factor of about 30x performance difference). The second goal is to clean up the code and allow for more interesting usage scenarios in the future.

## Installation and running your first MCMC

Please follow these steps:

1.  Clone this repository and `cd` into it:

    ```bash
    git clone https://github.com/parth7stark/MMA_RadioWave/tree/refit
    cd ./MMA_RadioWave/FederatedFitting
    ```

2. Create a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
   
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that the Octopus servers in `octopus-conf.sh` are correct.

5. Load the Octopus configuration:
    
    ```bash
    source ./otopus-conf.sh
    ```

5. Update server configuration file in `example/configs/server.yaml` if desired.
6. Update site configuration file in `example/configs/site1.yaml` if desired.

7. Start the site:
    
    ```bash
    PYTHONPATH=src python bin/run_site.py
    ```
8. Start the server:
    
    ```bash
    PYTHONPATH=src python bin/run_server.py
    ```
