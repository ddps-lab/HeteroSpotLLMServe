from typing import List, Dict, Tuple
import socket
import ray
import subprocess
import logging
import time

class Cluster:
    def __init__(self):
        if not ray.is_initialized():
            ray.init(address="auto")
        self.pipelines: List[Pipeline] = []
        
    def create_pipeline(self, node_layer_mapping: List[Tuple[str, int]], model_name: str, tensor_store_port: int, api_server_port: int):
        pipeline = Pipeline()
        pipeline.initialize_pipeline(node_layer_mapping, model_name)
        self.pipelines.append(pipeline)

class Pipeline:
    def __init__(self):
        self.vnodes: List[VNode] = []

    def initialize_pipeline(self, 
                            node_layer_mapping: List[Tuple[str, int]], 
                            model_name: str,
                            tensor_store_port: int,
                            api_server_port: int):
        assert len(node_layer_mapping) > 0, "node_layer_mapping is empty"

        if not ray.is_initialized():
            ray.init(address="auto")
        
        start_layer_idx = 0
        for pipeline_rank, (node_ip, layer_partition) in enumerate(node_layer_mapping):
            num_gpu = None
            while True:
                for node_info in ray.nodes():
                    ray_node_ip = node_info.get("NodeManagerAddress")
                    if ray_node_ip == node_ip:
                        num_gpu = int(node_info.get("Resources").get("GPU"))
                        break
                if num_gpu is not None:
                    break
                logging.info(f"Waiting for node {node_ip} to be entered into Ray cluster...")
                time.sleep(1)

            vnode = VNode(node_ip, num_gpu, pipeline_rank,
                          start_layer_idx, start_layer_idx + layer_partition)
            start_layer_idx += layer_partition
            self.vnodes.append(vnode)
        
        for vnode in self.vnodes:
            vnode.start_tensor_store(tensor_store_port, model_name)
            vnode.start_api_server(api_server_port, model_name)
        
        tensor_store_statuses = [False] * len(self.vnodes)
        api_server_statuses = [False] * len(self.vnodes)
        while not (all(tensor_store_statuses) and all(api_server_statuses)):
            for i, vnode in enumerate(self.vnodes):
                tensor_store_statuses[i] = vnode.check_tensor_store_status()
                api_server_statuses[i] = vnode.check_api_server_status()
            time.sleep(3)


class VNode:
    def __init__(self, 
                 node_ip: str, 
                 num_gpu: int,
                 pipeline_rank: int,
                 layer_start_id: int,
                 layer_end_id: int):
        self.node_ip = node_ip
        self.num_gpu = num_gpu
        self.pipeline_rank = pipeline_rank
        self.layer_start_id = layer_start_id
        self.layer_end_id = layer_end_id
        
        self.tensor_store_port = None
        self.is_tensor_store_ready = False
        self.tensor_store_process = None

        self.is_api_server_tcp_port = None
        self.is_api_server_ready = False
        self.api_server_process = None

    def start_tensor_store(self, tensor_store_port: int, model_name: str):
        self.tensor_store_port = tensor_store_port
        self.is_tensor_store_ready = False

        # TODO: Implement get_tensor_store_command
        tensor_store_command = get_tensor_store_command(self.node_ip, self.tensor_store_port, model_name)
        self.tensor_store_process = subprocess.Popen(
            tensor_store_command
        )
    
    def start_api_server(self, api_server_port: int, model_name: str):
        self.is_api_server_tcp_port = api_server_port
        self.is_api_server_ready = False

        # TODO: Implement get_api_server_command
        api_server_command = get_api_server_command(self.node_ip, self.api_server_port, model_name)
        self.api_server_process = subprocess.Popen(
            api_server_command
        )

    def check_tensor_store_status(self, timeout: float = 2.0) -> bool:
        try:
            with socket.create_connection((self.node_ip, self.tensor_store_port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                resp = sock.recv(4)  # '1' 또는 '0'
                status = resp.strip() == b"1"
                self.is_tensor_store_ready = status
                return status
        except (socket.timeout, ConnectionRefusedError, OSError):
            # 연결 실패 또는 응답 지연 : 준비되지 않은 것으로 간주
            self.is_tensor_store_ready = False
            return False

    def check_api_server_status(self, timeout: float = 2.0) -> bool:
        # TODO
        return True



if __name__ == "__main__":
    # TODO: Unit test
    pass