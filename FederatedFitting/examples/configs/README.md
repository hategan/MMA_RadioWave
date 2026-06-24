The format of the config files has changed: `server.yaml` and
`site1.yaml` use the new format. The others are kept for reference.
The main changes are:

* Server **and** site configuration files:

  * The options `run_name`, `plot_names`, `save_folder`, and various logging path names under `fitting_configs` have been replaced with a single `results_dir` global option (`logging_dir` for sites) which points to a directory that will contain pre-determined file names for logs, data, and various plots.
  * The section `comm_configs` has been removed. The Octopus/Kafka topics are now derived from the global option `cluster_id`. Kafka groups are not (at this time) used, and, until such time that consensus MCMC comes back, and if somehow we need to send more than 131,072 doubles in one message, we may add ProxyStore or some other solution back.
  * Dataset flags and loading options (`exclude_time_flag`, `exclude_ra_dec_flag`, `exclude_name_flag`, `exclude_uncertainty_flag`, `max_acceptable_flagnum`, `exclude_name_flag`, `target_name`, `name_of_source`,
   `GPS_time_of_GW`, `ra`, `dec`, `exclude_outside_ra_dec_uncertainty`,
   `arcseconds_uncertainty`) have been moved to the sites under `dataset`.
* Server changes:
  * An `expected_site_count` main option has been added. For now. It tells the server to expect these many sites to respond when starting an MCMC analysis.
  * `mcmc_configs` has been renamed to `mcmc`.
  * `mcmc.flux_model` can be used to select between `afterglowpy` and `fiesta`.
  * `mcmc.threads` can be used to select the number of OS/Python threads to run the MCMC simulation with. Experiments show that performance comparable with plain emcee is obtained for a number of threads of at least 64 and a number of walkers of at least 512.
  * The `<param>_known`, `<param>_fixed`, `<param>_range` scheme has been replaced with a simple `<param>` key. If the value is a range (e.g., `[0.1, 0.9]`), the parameter becomes an MCMC parameter. If the value is a sharp value (e.g., `0.3`), the parameter is considered known and is not varied during the MCMC run.
* Site configuration file changes:
  * The `mcmc_configs` section has been removed. The MCMC configuration is sent by the server with each run.
  * There is only one output configuration: `logging_dir`.
  * `client_id` has been removed; the client ID is generated automatically.
  * `cluster_id` has been added and must match the cluster ID of the server, the basic idea being that multiple independent clusters can be created concurrently (as well as multiple independent runs, but that part is handled automatically).
  * Dataset loading options (see above) have been moved from the server configuration under the `dataset` section.
  * The dataset path is now `dataset.path` rather than `fitting_configs.dataset_path`.