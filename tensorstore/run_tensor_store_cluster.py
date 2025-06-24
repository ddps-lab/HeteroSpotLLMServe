import argparse
import time
import sys
import os
import json
import subprocess
import shlex # shlex 모듈 임포트
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)
# 현재 파일의 절대 경로를 가져옵니다.
current_file_path = os.path.abspath(__file__)
# 부모 디렉터리 (tensorstore 폴더의 부모)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
# sys.path에 부모 디렉터리를 추가하여 utils.py를 찾을 수 있도록 합니다.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import run_server_remote

PYTHON = "/home/ubuntu/.conda/envs/vllm/bin/python"
TENSOR_SERVER_FILE = "/home/ubuntu/vllm-example/tensorstore/mt_tensor_store_server.py"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--node-rank-mapping-path", type=str, required=True)
    parser.add_argument("--pp-partition", type=str, required=True)
    args = parser.parse_args()

    node_rank_mapping = json.load(open(args.node_rank_mapping_path))
    # pp-partition 은 , 로 구분된 int 리스트임.
    pp_partition = [int(x) for x in args.pp_partition.split(",")]

    # 디버깅을 위한 print 문 추가
    print(f"DEBUG: len(node_rank_mapping) = {len(node_rank_mapping)}")
    print(f"DEBUG: len(pp_partition) = {len(pp_partition)}")
    print(f"DEBUG: node_rank_mapping path = {args.node_rank_mapping_path}")
    print(f"DEBUG: pp_partition input = {args.pp_partition}")

    # json 의 key 수 (node 수) 와 pp-partition 의 length 가 같아야 함.
    assert len(node_rank_mapping) == len(pp_partition), "node 수와 pp-partition 의 length 가 같아야 함."

    layer_idx = 0
    managed_tensor_store_servers: Dict[str, List[Optional[subprocess.Popen]]] = {}
    for node_ip, rank_list in node_rank_mapping.items():
        managed_tensor_store_servers[node_ip] = []
        tensor_parallel_size = len(rank_list)
        for local_rank in range(tensor_parallel_size):
            base_server_command = (
                f"{PYTHON} "
                f"{TENSOR_SERVER_FILE} "
                f"--model-name={args.model_name} "
                f"--tensor-parallel-size={tensor_parallel_size} "
                f"--local-rank={local_rank} "
                f"--start-layer-id={layer_idx} "
                f"--end-layer-id={layer_idx + pp_partition[local_rank]}"
            )

            hostname = node_ip
            username = "ubuntu"
            current_file_dir = os.path.dirname(current_file_path)
            log_file_path = os.path.join(current_file_dir, f"{node_ip}_{local_rank}.log")

            print(f"[{hostname}] 원격 서버 실행 명령어: {base_server_command}")
            print(f"[{hostname}] 원격 서버 로그 파일 경로: {log_file_path}")

            # utils.py의 run_server_remote를 호출한다고 가정 (이전 command.py의 내용과 유사)
            success, process = run_server_remote(hostname, username, base_server_command, log_file_path)
            if not success:
                print(f"[{hostname}] 원격 서버 실행 실패: {process}")
                sys.exit(1)
            else:
                print(f"[{hostname}] 원격 서버 실행 성공: {process}")

            managed_tensor_store_servers[node_ip].append(process)
        layer_idx += pp_partition[local_rank]

    try:
        while True:
            if not managed_tensor_store_servers:
                print("관리할 원격 프로세스가 없습니다. 5초 후 재확인...")
                time.sleep(5)
                if not managed_tensor_store_servers: # 다시 한번 확인 후 종료 결정
                    print("관리할 프로세스가 없어 루프를 종료합니다.")
                    break

            for hostname, proc_obj_list in managed_tensor_store_servers.items(): # dict 변경 중 순회를 위해 list 사용
                for proc_obj in proc_obj_list:
                    return_code = proc_obj.poll()

                    if return_code is None:
                        print
                        # 여기에 원격 로그 파일 확인 로직 추가 가능
                        # 예: check_remote_log(hostname, username, log_file_path_for_this_host)
                    else:
                        print(f"[{hostname}] ({time.strftime('%H:%M:%S')}) 원격 서버가 종료되었거나 연결이 끊어졌습니다! 반환 코드: {return_code}")
                        try:
                            stdout_data, stderr_data = proc_obj.communicate(timeout=1) # 이미 종료되었으므로 짧은 타임아웃
                            if stderr_data:
                                print(f"[{hostname}] SSH STDERR: {stderr_data.decode().strip()}")
                            if stdout_data: # 거의 없을 것으로 예상
                                print(f"[{hostname}] SSH STDOUT: {stdout_data.decode().strip()}")
                        except Exception as e:
                            print(f"[{hostname}] 종료된 SSH 프로세스로부터 stderr 읽기 실패: {e}")
                    
                    time.sleep(1)

                time.sleep(15) # 15초 간격으로 각 노드 프로세스 상태 확인

    except KeyboardInterrupt:
        logger.info("\nCtrl+C 입력 감지. 실행 중인 모든 원격 프로세스를 강제 종료 시도합니다...")
        
        # 종료해야 할 원격 호스트 및 해당 로컬 Popen 객체들 수집
        hosts_and_procs: Dict[str, List[subprocess.Popen]] = {}
        for hostname, proc_obj_list in managed_tensor_store_servers.items():
            active_local_procs = [p for p in proc_obj_list if isinstance(p, subprocess.Popen) and p.poll() is None]
            if active_local_procs:
                hosts_and_procs[hostname] = active_local_procs

        if not hosts_and_procs:
            logger.info("종료할 활성 원격 프로세스(또는 SSH 연결)가 없는 것 같습니다.")
        else:
            username = "ubuntu" # utils.py 와 동일하게 가정
            # pkill 패턴: TENSOR_SERVER_FILE 경로와 --model-name 인자를 포함하는 프로세스
            # shlex.quote를 사용하여 패턴 내 특수문자(공백 등)를 안전하게 처리
            pkill_pattern_unquoted = f"{TENSOR_SERVER_FILE}.*--model-name={args.model_name}"
            quoted_pkill_pattern = shlex.quote(pkill_pattern_unquoted)
            pkill_command = f"pkill -SIGKILL -f {quoted_pkill_pattern}"

            logger.info("원격 프로세스 강제 종료 (SIGKILL) 시도...")
            for hostname in hosts_and_procs.keys():
                try:
                    logger.info(f"[{hostname}] 원격 SIGKILL 명령 실행: ssh {username}@{hostname} '{pkill_command}'")
                    # check=False: pkill이 대상을 못찾아도 오류 발생 안함 (최선 노력)
                    # timeout: 응답이 너무 길어지는 것을 방지
                    subprocess.run(
                        ["ssh", f"{username}@{hostname}", pkill_command],
                        timeout=10, check=False, capture_output=True, text=True
                    )
                    logger.info(f"[{hostname}] 원격 SIGKILL 명령 전송 완료 (결과는 확인하지 않음).")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[{hostname}] 원격 SIGKILL 명령 시간 초과.")
                except Exception as e_pkill:
                    logger.error(f"[{hostname}] 원격 SIGKILL 명령 실행 중 오류: {e_pkill}")
            
            logger.info("원격 프로세스에 대한 강제 종료(SIGKILL) 시도 완료.")

            # 로컬 SSH 클라이언트 프로세스 정리 (최선 노력)
            logger.info("로컬 SSH 클라이언트 프로세스 정리 시도...")
            for hostname, proc_obj_list_for_host in hosts_and_procs.items():
                 for proc_obj in proc_obj_list_for_host:
                    if proc_obj.poll() is None:
                        try:
                            logger.info(f"[{hostname}] 로컬 SSH 클라이언트 (PID {proc_obj.pid}) 강제 종료(kill).")
                            proc_obj.kill() 
                            proc_obj.wait(timeout=2) # kill 후 짧게 대기
                        except Exception as e_kill_local:
                            logger.error(f"[{hostname}] 로컬 SSH 클라이언트 (PID {proc_obj.pid}) kill 중 오류: {e_kill_local}")
                    # else: 이미 종료된 경우 특별히 로깅 안함
        
        logger.info("모든 종료 처리 시도 완료.")

    finally:
        print("스크립트 종료.")