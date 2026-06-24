import logging
import threading
import time
import uuid
from concurrent.futures import Future
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, Tuple, Set
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from mma_fedfit.server.octopus_server_communicator import OctopusServerCommunicator
import json
import emcee

from mma_fedfit.server.plots import Plots

logger = logging.getLogger(__name__)


MAX_RESULT_TIME_MS = 30_000

# These are parameters as they must appear in the configuration file.
PARAMS = ['thetaObs', 'thetaCore', 'thetaWing', 'p', 'logE0', 'logn0', 'xiN', 'logEpsilon_e',
          'logEpsilon_B', 'z', 'DL']
PARAM_DEFAULTS = {
    'jetType': 0,
    'specType': 0,
    'counterjet': True,
    'xiN': 1.0
}


# The general idea is as follows:
#   * A vitrual cluster is created by setting a "cluster_id" in the configuration file
#   * Quasi-persistent sites are then started and listen to the cluster topic, which is created
#     if not already there
#   * When a server starts, it creates a run with a unique ID and a topic for that run. The server
#     then advertises on the cluster topic that a run has been created and collects the identities
#     of all the responding sites. Sites respond to the run started message on the run topic.
#   * The server the proceeds to request log likelihoods from sites on the cluster topic and
#     receives responses on the run topic.
#   * When a run completes, the run topic is deleted by the server.


class Replies:
    def __init__(self, sites: Set[str], call_id: str) -> None:
        self.sites = set(sites)
        self.call_id = call_id
        self.sum = 0.0
        self.future = Future()

    def result_received(self, site_id: str, value: float) -> None:
        if not site_id in self.sites:
            logger.warning(f'Received spurious reply from {site_id} for call {self.call_id}')
            return
        self.sites.remove(site_id)
        self.sum += value
        if len(self.sites) == 0:
            self.future.set_result(self.sum)


class Server:
    def __init__(self, config_file: str) -> None:
        self.config = OmegaConf.load(config_file)
        self.run_id = uuid.uuid1().hex
        self.bus = OctopusServerCommunicator(self.config, self.run_id)
        self.site_ids = set()
        self.fixed_params: Dict[str, object] = {}
        self.ranges: Dict[str, Tuple[float, float]] = {}
        self.lower_bounds = None
        self.upper_bounds = None
        self.theta_params = []
        self.nwalkers = self.config.mcmc.nwalkers
        self.call_id = 0
        self.lock = threading.RLock()
        self.replies: Dict[str, Replies] = {}
        self.receiver_thread = threading.Thread(target=self.receiver)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        self.nn = 0
        self.sn = 0.0
        self.an = 0.0
        self.start_time = 0

    def run(self) -> None:
        self.process_params()
        self.discover_sites()
        sampler = self.run_mcmc()
        self.process_results(sampler)

    def run_mcmc(self) -> emcee.EnsembleSampler:
        niters = self.config.mcmc.niters
        ndim = len(self.ranges)
        pos = np.random.uniform(low=self.lower_bounds, high=self.upper_bounds,
                                size=(self.nwalkers, ndim))

        self.start_time = time.time()
        with ThreadPool(processes=self.config.mcmc.threads) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers, ndim, self.global_log_probability, pool=pool,
                moves=[(emcee.moves.StretchMove(a=1.1), 0.7), (emcee.moves.WalkMove(10), 0.3)])

            sampler.run_mcmc(pos, niters, progress=True)
            return sampler

    def get_Z(self, theta: np.array) -> Dict[str, float]:
        assert len(self.theta_params) == len(theta)
        Z = {}
        for i in range(len(self.theta_params)):
            Z[self.theta_params[i]] = theta[i]
        return Z

    def global_log_probability(self, theta) -> float:
        if not np.all((self.lower_bounds <= theta) & (theta <= self.upper_bounds)):
            return -np.inf

        with self.lock:
            call_id = self.call_id
            self.call_id += 1

        data = {
            'EventType': 'ProposedTheta',
            'RunId': self.run_id,
            'CallId': call_id,
            'Z': self.get_Z(theta)
        }
        r = Replies(self.site_ids, call_id)
        with self.lock:
            self.replies[call_id] = r
            self.nn += 1

        self.bus.send_to_cluster(data)

        return r.future.result()

    def receiver(self) -> None:
        print('Receiver started')
        for data in self.bus.run_collect():
            assert 'EventType' in data
            assert data['EventType'] == 'PartialLogLikelihoodResult'
            call_id = data['CallId']
            site_id = data['SiteId']
            value = data['Likelihood']
            with self.lock:
                if not call_id in self.replies:
                    logger.warning('Received reply for unknown call: %s', call_id)
                r = self.replies[call_id]
            r.result_received(site_id, value)

    def process_params(self) -> None:
        lb = []
        ub = []
        for name in PARAMS:
            if name not in self.config.mcmc:
                if name not in PARAM_DEFAULTS:
                    raise KeyError(f'Parameter {name} not found in configuration file and not '
                                   f'default is available.')
                value = PARAM_DEFAULTS[name]
                print(f'Parameter {name} not found in configuration file. Using default value of '
                      f'{value}')
                self.fixed_params[name] = value
            else:
                cv = self.config.mcmc[name]
                if isinstance(cv, omegaconf.ListConfig):
                    # got a range
                    self.ranges[name] = (cv[0], cv[1])
                    lb.append(cv[0])
                    ub.append(cv[1])
                    self.theta_params.append(name)
                else:
                    if isinstance(cv, int):
                        cv = float(cv)
                    if not isinstance(cv, float):
                        raise ValueError(f'Expected float for {name} but got type {type(cv)}.')
                    self.fixed_params[name] = cv

        self.lower_bounds = np.array(lb)
        self.upper_bounds = np.array(ub)

    def discover_sites(self) -> None:
        # we publish a RunStarted event and wait for a while to hear all
        # reporting sites, then run MCMC with those sites
        message = {
            'EventType': 'RunStarted',
            'RunId': self.run_id,
            'ServerConfig': OmegaConf.to_container(self.config, resolve=True),
            'FixedParams': self.fixed_params,
            'MinTime': 0,
            'MaxTime': 99999999
        }
        self.bus.send_to_cluster(message)
        expected_site_count = self.config.expected_site_count
        for data in self.bus.cluster_collect(time_ms = 10_000):
            assert 'EventType' in data
            if data['EventType'] == 'RunStarted':
                # our very own message
                continue
            assert data['EventType'] == 'SiteReady'
            site_id = data['SiteId']
            self.site_ids.add(site_id)
            print(f'Received response from site {site_id}')
            if expected_site_count == len(self.site_ids):
                break
        print(f'{len(self.site_ids)} sites responded')
        if len(self.site_ids) == 0:
            raise RuntimeError(f'No sites have responded out of {expected_site_count} '
                               f'expected. Please make sure that the sites are running.')

    def process_results(self, sampler: emcee.EnsembleSampler) -> None:
        results_dir = self.config.results_dir
        run_dir = Path(results_dir) / f'run_{self.run_id}'
        run_dir.mkdir(parents=True, exist_ok=True)

        full_chain = sampler.get_chain(discard=0, flat=True)
        np.save(run_dir / 'full_chain.npy', full_chain)

        burn_in = self.config.mcmc.burnin
        samples = sampler.get_chain(discard=burn_in, flat=True)

        end = time.time()
        total_time = end - self.start_time
        spsf = len(full_chain) / total_time
        sps = len(samples) / total_time

        print(f'Stats: nwalkers={self.nwalkers}, nthreads={self.config.mcmc.threads}, '
              f'nsamples={len(full_chain)}, time={total_time:.3f}, sps={spsf:.3f}, effective_sps={sps:.3f}')

        np.save(run_dir / 'with_burnin.npy', samples)
        np.save(run_dir / 'log_prob.npy', sampler.get_log_prob(discard=burn_in, flat=True))
        np.save(run_dir / 'log_prob_full.npy', sampler.get_log_prob(flat=False))

        plots = Plots(run_dir, self.theta_params, samples)
        plots.make_plots()
