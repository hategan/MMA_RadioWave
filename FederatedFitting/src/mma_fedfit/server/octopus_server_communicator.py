import json

import time
from datetime import timedelta
from typing import Dict, Generator, Optional, Any
from diaspora_event_sdk import Client, KafkaProducer, KafkaConsumer
import logging
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


class OctopusServerCommunicator:
    def __init__(self, config: OmegaConf, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.cluster_id = config.cluster_id

        client = Client()
        self.cluster_topic = f'{client.namespace}.{self.cluster_id}'
        self.run_topic = f'{client.namespace}.{self.run_id}'
        self.dcall(client.create_topic, self.cluster_id)
        self.dcall(client.create_topic, self.run_id)

        self.cluster_producer = KafkaProducer(self.cluster_topic)
        self.cluster_consumer = KafkaConsumer(self.cluster_topic, enable_auto_commit=True,
                                              auto_offset_reset='latest')

    def dcall(self, c, *args):
        response = c(*args)
        if 'status' in response and response['status'] != 'success':
            raise RuntimeError(response['message'])

    def send_to_cluster(self, data: Dict[str, object], flush: bool = False) -> None:
        self.cluster_producer.send(self.cluster_topic, data)
        if flush:
            self.cluster_producer.flush()

    def flush_cluster(self) -> None:
        self.cluster_producer.flush()

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

    def print_error(self, data: Dict[str, str]) -> None:
        site_id = data['SiteId'] if 'SiteId' in data else '-'
        run_id = data['RunId'] if 'RunId' in data else '-'
        err = data['Error']
        trace = data['Details']
        print(f'Site {site_id}, run {run_id} error: {err}\n{trace}')

    def cluster_collect(self, time_ms: int = 0) -> Generator[Dict[Any, Any], None, None]:
        yield from self._collect(self.cluster_consumer, time_ms)

    def run_collect(self, time_ms: int = 0) -> Generator[Dict[Any, Any], None, None]:
        run_consumer = KafkaConsumer(self.run_topic, enable_auto_commit=True,
                                     auto_offset_reset='latest')
        yield from self._collect(run_consumer, time_ms)
