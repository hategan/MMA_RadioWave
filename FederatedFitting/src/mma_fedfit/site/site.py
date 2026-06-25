import logging
import traceback
import uuid
from datetime import timedelta, datetime
from multiprocessing import Manager, Queue
from typing import Tuple, Dict

from omegaconf import OmegaConf

from .data import DataProcessor
from mma_fedfit.models import get_model
from .octopus_client_communicator import OctopusClientCommunicator

import pandas as pd
import numpy as np
import threading

logger = logging.getLogger(__name__)


MAX_RUN_AGE = timedelta(days=4)


def parse_td(td: str | int) -> timedelta:
    """
    Parses a time interval.

    The format is "<int><spaces>*['s', 'm', 'h', 'd']". That is some
    integer followed by one of the suffixes that stand for seconds, minutes,
    hours, or days, with optional spaces in-between.

    :param td: A string representation of the time interval in the above format.
    :return: A timedelta object.
    """
    if isinstance(td, int):
        return timedelta(seconds=td)
    assert isinstance(td,str)
    suffix = td.strip()[-1]
    num = int(td.strip()[:-1].strip())
    if suffix == 's':
        return timedelta(seconds=num)
    if suffix == 'm':
        return timedelta(minutes=num)
    if suffix == 'h':
        return timedelta(hours=num)
    if suffix == 'd':
        return timedelta(days=num)


class Likelihood:
    def __init__(self, t, nu, fnu, err, fixed_params, model):
        self.t = t
        self.nu = nu
        self.fnu = fnu
        self.err = err
        self.fixed_params = fixed_params
        self.model = model

    def log_likelihood(self, Z: Dict[str, float]) -> float:
        Z.update(self.fixed_params)
        fnu = self.model.flux_density(self.t, self.nu, Z)
        sigma2 = self.err ** 2
        return -0.5 * np.sum((fnu - self.fnu) ** 2 / sigma2 + np.log10(sigma2))


class Run:
    def __init__(self, run_id: str, config: Dict[str, object], data, limits,
                 min_time: timedelta, max_time: timedelta,
                 fixed_params: Dict[str, object]) -> None:
        self.run_id = run_id
        self.config = config

        min_seconds = min_time.total_seconds()
        max_seconds = max_time.total_seconds()
        self.data = data[(min_seconds <= data['t']) & (data['t'] <= max_seconds)]
        self.fixed_params = fixed_params

        self.n_pts = len(self.data)

        t = np.array(self.data["t"])
        nu = np.array(self.data["frequency"])
        fnu = np.array(self.data["flux"])
        err = np.array(self.data["err"])
        model = get_model(config['mcmc']['flux_model'])
        self.likelihood = Likelihood(t, nu, fnu, err, fixed_params, model)
        self.created = datetime.now()


def _compute_and_send(call_id: str, run_id: str, likelihood: Likelihood, Z: Dict[str, float],
                      queue: Queue) -> None:
    try:
        ll = likelihood.log_likelihood(Z)
        queue.put((run_id, call_id, ll))
    except Exception as ex:
        traceback.print_exc()
        queue.put((run_id, call_id, ex))


class Site:
    def __init__(self, config_file: str) -> None:
        self.config = OmegaConf.load(config_file)
        self.site_id = uuid.uuid1().hex
        self.bus = OctopusClientCommunicator(self.config, self.site_id)
        self.mp_manager = Manager()
        self.queue = self.mp_manager.Queue()
        self.pool = self.mp_manager.Pool(self.config.max_threads)
        self.lock = threading.RLock()
        self.runs = {}
        self.sender = threading.Thread(target=self.sender)
        self.sender.daemon = True
        self.sender.start()

    def read_data(self) -> Tuple[object, object]:
        data_dir = self.config.dataset.path
        dataset_name = data_dir.split('/')[-1].split('_')[0]

        # print(f"Computing log-likelihood on {dataset_name} dataset", flush=True)
        logger.info(f'[Site {self.site_id}] reading {dataset_name}')

        # Load flux-time at the site (local data)
        raw_data = pd.read_csv(data_dir)

        if raw_data.shape[1] == 1:
            raw_data = pd.read_csv(data_dir, delim_whitespace=True)

        # Preprocess local data
        dp = DataProcessor(raw_data, self.config)
        data = dp.interpret()
        limits = dp.interpret_ULs()
        return data, limits

    def run(self) -> None:
        self.data, self.limits = self.read_data()
        print(f'Dataset loaded with {len(self.data)} points.')
        self.process_messages()

    def run_started(self, run_data: Dict[str, object]) -> None:
        try:
            run_id = run_data['RunId']
            self.bus.add_run(run_id)
            print(f'Run {run_id} started.')

            try:
                server_config = run_data['ServerConfig']
                fixed_params = run_data['FixedParams']
                run = Run(run_id, server_config, self.data, self.limits,
                          parse_td(run_data['MinTime']), parse_td(run_data['MaxTime']),
                          fixed_params)
                with self.lock:
                    if run_id in self.runs:
                        raise KeyError(f'Run {run_id} already registered.')
                    self.runs[run_id] = run

                self.purge_old_runs()

                self.send_site_ready(run_id, run.n_pts)
            except Exception as ex:
                traceback.print_exc()
                self.send_run_error(run_id, ex)
        except Exception as ex:
            traceback.print_exc()
            self.send_cluster_error(ex)

    def send_run_error(self, run_id: str, ex: Exception) -> None:
        response = {
            'EventType': 'Error',
            'SiteId': self.site_id,
            'RunId': run_id,
            'Error': str(ex),
            'Details': ''.join(traceback.format_exception(ex))
        }
        self.bus.send_to_run(run_id, response)

    def send_cluster_error(self, ex: Exception) -> None:
        response = {
            'EventType': 'Error',
            'SiteId': self.site_id,
            'Error': str(ex),
            'Details': ''.join(traceback.format_exception(ex))
        }
        self.bus.send_to_cluster(response)

    def purge_old_runs(self) -> None:
        to_remove = []
        now = datetime.now()
        for run in self.runs.values():
            if now - run.created > MAX_RUN_AGE:
                to_remove.add(run)
        with self.lock:
            for run in to_remove:
                del self.runs[run.run_id]
        for run in to_remove:
            self.bus.remove_run(run.run_id)

    def send_site_ready(self, run_id: str, n_pts: int) -> None:
        response = {
            'EventType': 'SiteReady',
            'SiteId': self.site_id,
            'NPts': n_pts
        }
        self.bus.send_to_cluster(response)

    def partial_log_likelihood(self, data: Dict[str, object]) -> None:
        # We use a single server -> site topic for all runs because it really
        # doesn't matter: we respond the same way to all log likelihood
        # requests. On the other hand, we use a separate topic for the
        # responses because each server instance is bound to a different
        # run.
        run_id = data['RunId']
        if not run_id in self.runs:
            return
        call_id = data['CallId']
        with self.lock:
            run = self.runs[run_id]
        Z = data['Z']
        logger.debug('Site %s received proposed theta: call_id: %s"',
                     self.site_id, call_id)
        self.pool.apply_async(_compute_and_send, (call_id, run.run_id, run.likelihood, Z, self.queue))

    def sender(self) -> None:
        print('Sender started')
        while True:
            run_id, call_id, value = self.queue.get()
            if isinstance(value, Exception):
                self.send_run_error(run_id, value)
            else:
                response = {
                    'EventType': 'PartialLogLikelihoodResult',
                    'SiteId': self.site_id,
                    'CallId': call_id,
                    'Likelihood': value
                }
                self.bus.send_to_run(run_id, response)

    def process_messages(self) -> None:
        print('Waiting for messages.')
        n = 0
        s = 0.0
        for data in self.bus.cluster_collect():
            logger.info(f'[Site %s] msg: %s', self.site_id, data)

            assert 'EventType' in data
            event_type = data['EventType']

            if event_type == 'RunStarted':
                self.run_started(data)
            elif event_type == 'ProposedTheta':
                self.partial_log_likelihood(data)
            elif event_type == 'AggregationDone':
                # not doing anything for now; in principle, this could be used
                # to save results to local sites
                pass
            else:
                logger.info('[Site %s: Unknown event type: %s', self.site_id, data)


