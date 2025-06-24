import torch
import logging
from multiprocessing.managers import BaseManager, DictProxy
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [Client] - %(message)s')

MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001
MANAGER_AUTHKEY = b'temp_authkey'

class TensorManager(BaseManager):
    pass

# 매니저에 서버에서 등록한 함수 등록
# 클라이언트 측에서도 DictProxy를 사용하도록 지정합니다.
TensorManager.register('get_tensors', proxytype=DictProxy)


def main():
    shared_tensors = {}
    logging.info(f"Attempting to connect to TensorManager server at {MANAGER_HOST}:{MANAGER_PORT}...")

    # 매니저 클라이언트 생성
    manager = TensorManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=MANAGER_AUTHKEY)

    max_retries = 5
    retry_delay = 2 # 초
    for attempt in range(max_retries):
        try:
            # 서버에 연결 시도
            manager.connect()
            logging.info("Connected to TensorManager server.")
            break # 연결 성공
        except ConnectionRefusedError:
            logging.warning(f"Connection refused (Attempt {attempt + 1}/{max_retries}). Server might not be ready. Retrying in {retry_delay}s...")
            if attempt == max_retries - 1:
                logging.error("Max connection attempts reached. Exiting.")
                return
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Error connecting to manager: {e}")
            return # 다른 오류 발생 시 종료

    try:
        logging.info("Accessing shared tensors via manager...")
        # 매니저를 통해 공유 딕셔너리 프록시 가져오기
        shared_tensor_data_proxy = manager.get_tensors()

        # 프록시 딕셔너리의 내용을 로컬 딕셔너리로 복사 (필요한 경우)
        # 주의: 이렇게 복사하면 텐서 자체가 로컬 프로세스로 복사되는 것이 아니라,
        # 여전히 공유 메모리를 가리키는 프록시 텐서 객체가 복사됩니다.
        # shared_tensors = dict(shared_tensor_data_proxy) # 방법 1: 전체 복사

        # 또는 필요할 때마다 프록시를 통해 직접 접근
        tensor_names = list(shared_tensor_data_proxy.keys()) # 키 목록 먼저 가져오기
        logging.info(f"Received access to {len(tensor_names)} shared tensors.")

        if tensor_names:
            logging.info("Testing access to a few tensors...")
            test_names = tensor_names # 전체 텐서 검사

            logging.info("Performing integrity checks on accessed tensors...")
            successful_checks = 0
            failed_checks = 0
            for name in test_names:
                try:
                    # 프록시를 통해 텐서 접근
                    tensor = shared_tensor_data_proxy[name]

                    # --- 무결성 검사 수행 ---
                    actual_shape = tuple(tensor.shape)
                    actual_dtype = tensor.dtype
                    is_contiguous = tensor.is_contiguous()
                    norm_val = tensor.norm().item()

                    logging.debug(f"  Checking tensor '{name}'...")
                    logging.debug(f"    - Shape: {actual_shape}")
                    logging.debug(f"    - Dtype: {actual_dtype}")
                    logging.debug(f"    - Is Contiguous: {is_contiguous}")
                    logging.debug(f"    - Norm: {norm_val:.4f}")
                    logging.debug(f"    - Device: {tensor.device}")
                    # INFO 레벨에서는 성공/실패 요약만 남기도록 변경 가능
                    # logging.info(f"    -> Integrity checks passed for '{name}'.")
                    successful_checks += 1

                    # 로컬 딕셔너리에 저장 (선택 사항)
                    shared_tensors[name] = tensor

                except Exception as check_e:
                    logging.error(f"  Checking tensor '{name}'...")
                    logging.error(f"    -> FAILED integrity check or access for tensor '{name}': {check_e}")
                    failed_checks += 1
            logging.info(f"Finished integrity checks. Passed: {successful_checks}, Failed: {failed_checks} (Total: {len(test_names)}). Check DEBUG logs for details.")
        else:
            logging.warning("No tensors found in the shared data.")

        # 접근이 완료되었음을 명시적으로 알릴 수 있음 (여기서는 단순히 종료)
        logging.info("Finished accessing tensors.")

    except Exception as e:
        logging.exception(f"An error occurred while accessing shared tensors: {e}")

    logging.info("\nFinal check of locally held tensor references:")
    if shared_tensors:
        logging.info(f"  Successfully accessed and held references to {len(shared_tensors)} tensors.")
        if shared_tensors: # 리스트가 비어있지 않은지 확인
            first_tensor_name = list(shared_tensors.keys())[0]
            logging.info(f"  Example: First accessed tensor '{first_tensor_name}' (Device: {shared_tensors[first_tensor_name].device})")
    else:
        logging.info("  No tensors were successfully accessed or held locally.")

    logging.info("Finished.")


if __name__ == "__main__":
    main() 