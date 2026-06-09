import json
import logging
from typing import Optional
from omegaconf import OmegaConf
from proxystore.proxy import extract
from mma_fedfit.agent import ServerAgent
from mma_fedfit.logger import ServerAgentFileLogger
from .utils import serialize_tensor_to_base64, deserialize_tensor_from_base64

from diaspora_event_sdk import Client, KafkaProducer, KafkaConsumer
from proxystore.proxy import Proxy, extract
import threading
import json
import time
from collections import defaultdict
import sys
class OctopusServerCommunicator:
    """
    Octopus communicator for federated learning server.
    Contains functions to produce/consume/handle different events
    """

    def __init__(
        self,
        server_agent: ServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:

        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()

        topic = self.server_agent.server_agent_config.server_configs.comm_configs.octopus_configs.topic

        client = Client()
        client.create_topic(topic)

        # the Kafka topic is different from the Octopus topic
        self.topic = f'{client.namespace}.{topic}'

        # Kafka producer for publishing messages
        self.producer = KafkaProducer(self.topic, request_timeout_ms=3000000, delivery_timeout_ms=4000000, max_block_ms=3000000)

        # Kafka consumer to listen for control events AND embeddings
        self.consumer = KafkaConsumer(
            self.topic,
            enable_auto_commit=True,
            group_id=self.server_agent.server_agent_config.server_configs.comm_configs.octopus_configs.group_id
        )

        self.site_summary = {}

        # Track readiness, which detector is connected and ready for inference
        self.active_sites = set()
        self.expected_sites = set()


        # In‐memory store: (iteration, walker) → { site_id: likelihood }
        self.store: dict[tuple[int,int], dict[str,float]] = defaultdict(dict)
        self.lock = threading.Lock()
        # self._start_background_poller()

        # This is to handle scenario: You do not wait for all detectors to say “ready” if you want to begin listening to embeddings as soon as any client is ready.

    def publish_server_started_event(self, threshold):
        """
        Publishes an event to the control topic indicating that the server has started,
        along with the configuration shared among all clients.
        """
        client_config =  self.server_agent.get_client_configs()
        client_config_dict = OmegaConf.to_container(client_config, resolve=True)

        self.threshold = threshold
        event = {
            "EventType": "ServerStarted",
            "site_config": client_config_dict,
            "day_threshold": self.threshold,
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        
        # print("[Server] Published ServerStarted event with config.", flush=True)
        self.logger.info("[Server] Published ServerStarted event with config.")

    def send_aggregation_results(self, results_dict):
        """
        Publishes an event to the control topic indicating that the server has started,
        along with the configuration shared among all clients.
        """

        event = {
        
            "EventType": "AggregationDone",
            "status": "HAS_DATA",
            "theta_est": results_dict,
            "day_threshold": self.threshold,
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        
        # print("[Server] Published AggregationDone event with best parameter estimates.", flush=True)
        self.logger.info("[Server] Published AggregationDone event with best parameter estimates.")


    def handle_local_MCMC_done_message(self, data):
        """
        Message of type "LocalMCMCDone" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=7705, timestamp=1736957074944, timestamp_type=0, key=None, value=b'{"EventType": "PostProcess", "detector_id": "1", "status": "DONE", "details": "DONE -> Invoke post process pipeline", "GPS_start_time": 1264314069}', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=147, serialized_header_size=-1)
        """

        site_id = data["site_id"]    # 0 or 1
        status = data["status"]
        chain_b64 = data["chain"]
        # min_time = data["min_time"]
        # max_time = data["max_time"]
        # unique_frequencies = data["unique_frequencies"]

        if status == "SKIPPED":
            self.logger.info(f"[Server] Site {site_id} skipped MCMC — no data")
            local_chain_list = []
        else:
            # Deserialize and extract tensor
            local_tensor = deserialize_tensor_from_base64(chain_b64)

            if isinstance(local_tensor, Proxy):
                local_tensor = extract(local_tensor)

            local_chain_list = local_tensor.tolist()  # Get back list for further processing

        # print(f"[Site {site_id}] chains: {local_chain_list}", flush=True)
        # self.logger.info(f"[Site {site_id}] chains: {local_chain_list}")



        # self.server_agent.aggregator.process_local_MCMC_done_message(self.producer, self.topic, site_id, status, local_chain_list, min_time, max_time, unique_frequencies)
        self.server_agent.aggregator.process_local_MCMC_done_message(self.producer, self.topic, site_id, status, local_chain_list, self.site_summary)

    def handle_SiteReady_message(self, data, num_sites):
        """
        Message of type "SiteReady" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=7705, timestamp=1736957074944, timestamp_type=0, key=None, value=b'{"EventType": "PostProcess", "detector_id": "1", "status": "DONE", "details": "DONE -> Invoke post process pipeline", "GPS_start_time": 1264314069}', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=147, serialized_header_size=-1)
        """


        site_id = str(data["Site_id"])    # 0 or 1
        day_threshold = data["day_threshold"]
        has_data = data["has_data"]
        n_data_points = data["n_data_points"]
    
        
        self.site_summary[site_id] = {
                        "has_data":       has_data,
                        "n_data_points":   data.get("n_data_points", 0),
                        "day_threshold": data.get("day_threshold", 0),
                    }
        
        # self.expected_sites.add(site_id)
        if has_data:
            self.active_sites.add(site_id)
        self.logger.info(f"[Server] Site {site_id} ready -> has_data={has_data}, n_data_points={n_data_points}, day_threshold={day_threshold} ")
        
        # Detect when all sites are ready. check if there is any site which has data
        if len(self.site_summary) == num_sites:
            if len(self.active_sites)==0:
                self.logger.info("[Server] All sites responded — but none have usable data. Skipping MCMC.")
                # Optionally: broadcast dummy AggregationDone
                self.send_empty_aggregation_result()
        # return self.site_summary, self.active_sites

    def send_empty_aggregation_result(self):
        
        event = {
        
            "EventType": "AggregationDone",
            "status": "NO_DATA",
            "theta_est": "none",
            "day_threshold": self.threshold,
        }

        

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        self.logger.info("[Server] Broadcasted empty AggregationDone due to lack of data.")
        
        sys.exit(0)

    def _start_background_poller(self):
        self.logger.info(f"[Server] Started Background poller/event listener")

        thread = threading.Thread(target=self._background_poller, daemon=True)
        thread.start()

    def _background_poller(self):
        for msg in self.consumer:
            data = json.loads(msg.value.decode('utf-8'))
            if data.get("EventType") == "LogLikelihoodComputed":
                key = (data["iteration_no"], data["walker_no"])
                with self.lock:
                    # Store every reply for later
                    self.store[key][data["site_id"]] = data["local_likelihood"]


    def send_proposed_theta(self, theta, iteration_no, walker_no):

        # print("proposed_theta", flush=True)
        event = {
            "EventType": "ProposedTheta",
            "iteration_no": iteration_no,
            "walker_no": walker_no,
            "theta": theta.tolist(),  # Convert ndarray to list,
            "day_threshold": self.threshold,
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        # print(future.get(timeout=10))
        
        # print(f"[Server] Published ProposedTheta event. Iteration no: {iteration_no}, walker={walker_no}", flush=True)
        self.logger.info(f"[Server] Published ProposedTheta event. Iteration no: {iteration_no}, walker={walker_no}")

    def collect_local_likelihoods(self, num_sites, ongoing_iteration, walker_no):
        
        # print("collect likelihood", flush=True)

        # log_likelihoods = {}
        # received_sites = 0

        # Track which clients have finished
        # completed_clients = set()
        # expected_clients = {str(i) for i in range(num_sites)}

        key = (ongoing_iteration, walker_no)
        # Use active sites instead of num_sites
        # expected_sites = {str(i) for i in range(num_sites)}
        expected_sites = self.active_sites  # only sites with data

        # while received_sites < num_sites:
        #     for message in consumer:
        #         data_str = msg.value.decode("utf-8")  # decode to string
        #         data = json.loads(data_str)          # parse JSON to dict

        #         Event_type = data["EventType"]

        #         if Event_type == "LogLikelihoodComputed"
        #             log_likelihoods[message.value["site_id"]] = message.value["log_likelihood"]
        #             received_sites += 1
        #             print(f"Received log-likelihood from Site {message.value['site_id']}")
        #             if received_sites == num_sites:
        #                 break

        # for msg in self.consumer:
        #     data_str = msg.value.decode("utf-8")  # decode to string
        #     data = json.loads(data_str)          # parse JSON to dict

        #     Event_type = data["EventType"]

        #     if Event_type == "LogLikelihoodComputed":
        #         site_id = data["site_id"]
        #         local_likelihood = data["local_likelihood"]
        #         iteration_no_in_msg = data["iteration_no"]

        #         if iteration_no_in_msg == ongoing_iteration:
        #             print(f"[Server] Received LogLikelihoodComputed Event from site {site_id}")
        #             self.logger.info(f"[Server] Received LogLikelihoodComputed from site {site_id}")

        #             # Add the client to the completed set
        #             completed_clients.add(site_id)

        #             log_likelihoods[site_id] = local_likelihood

        #             if completed_clients == expected_clients:
        #                 print("[Server] Collected partial log likelihoods from all the sites. Compute global posterior...")
        #                 self.logger.info("[Server] Collected partial log likelihoods from all the sites. Compute global posterior...")
        #                 return log_likelihoods
                    
        # Block until all sites have replied
        while True:
            with self.lock:
                got = self.store.get(key, {})
                # if got.keys() >= expected_sites:
                if set(got.keys()) == expected_sites:

                    result = got.copy()
                    # Clean up so we don’t grow unbounded
                    del self.store[key]
                    # print(f"[Server] Collected all likelihoods for (iter={ongoing_iteration},walker={walker_no})", flush=True)
                    self.logger.info(f"[Server] Collected all likelihoods for (iter={ongoing_iteration},walker={walker_no})")
                    
                    return result
            time.sleep(0.01)


        # return log_likelihoods


    def _default_logger(self):
        """Create a default logger for the server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger


