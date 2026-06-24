import json
import threading
import time
from typing import Dict, Generator, Any
from diaspora_event_sdk import Client, KafkaProducer, KafkaConsumer
import logging
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


class OctopusClientCommunicator:
    """
    Octopus communicator for federated learning clients.
    Contains functions to produce/consume/handle different events
    """
    def __init__(self, config: OmegaConf, site_id: str) -> None:
        """        
        :param client_id: A unique client ID.
        :param max_message_size: The maximum message size in bytes.
        """
        self.site_id = site_id
        self.cluster_id = config.cluster_id
        self.config = config
        self.run_topics: Dict[str, str] = {}
        self.run_producers: Dict[str, KafkaProducer] = {}
        self.lock = threading.RLock()

        self.client = Client()
        self.cluster_topic = f'{self.client.namespace}.{self.cluster_id}'
        self.dcall(self.client.create_topic, self.cluster_id)
        self.cluster_producer = KafkaProducer(self.cluster_topic)
        self.cluster_consumer = KafkaConsumer(self.cluster_topic, enable_auto_commit=True,
                                              auto_offset_reset='latest')

    def dcall(self, c, *args):
        response = c(*args)
        if 'status' in response and response['status'] != 'success':
            raise RuntimeError(response['message'])

    def send_to_cluster(self, data: Dict[str, object], flush: bool = True) -> None:
        self.cluster_producer.send(self.cluster_topic, data)
        if flush:
            self.cluster_producer.flush()

    @property
    def cluster_messages(self):
        return self.cluster_consumer

    def _collect(self, consumer: KafkaConsumer, time_ms: int) -> Generator[str, None, None]:
        if time_ms == 0:
            deadline_ms = time.time() * 1000 + 1_000_000_000
        else:
            deadline_ms = time.time() * 1000 + time_ms
        while True:
            now_ms = time.time() * 1000
            left_ms = deadline_ms - now_ms
            if left_ms < 0:
                return
            for partition, record_list in consumer.poll(timeout_ms=left_ms).items():
                for record in record_list:
                    data = json.loads(record.value.decode('utf-8'))
                    if 'EventType' in data and data['EventType'] == 'Error':
                        self.print_error(data)
                        continue
                    yield data

    def cluster_collect(self, time_ms: int = 0) -> Generator[Dict[Any, Any], None, None]:
        yield from self._collect(self.cluster_consumer, time_ms)

    def add_run(self, run_id: str) -> None:
        run_topic = f'{self.client.namespace}.{run_id}'
        producer = KafkaProducer(run_topic)
        with self.lock:
            self.run_topics[run_id] = run_topic
            self.run_producers[run_id] = producer

    def remove_run(self, run_id: str) -> None:
        with self.lock:
            del self.run_topics[run_id]
            del self.run_producers[run_id]

    def send_to_run(self, run_id: str, data: Dict[str, object], flush: bool = False):
        with self.lock:
            topic = self.run_topics[run_id]
            producer = self.run_producers[run_id]
        producer.send(topic, data)
        if flush:
            producer.flush()