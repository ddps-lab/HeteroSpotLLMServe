import torch
from transformers import LlamaForCausalLM
import time
import logging
from multiprocessing.managers import BaseManager, DictProxy
import torch.multiprocessing as mp

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [Server] - %(message)s')

# Manager 설정
MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001 # 포트 변경 (기존 소켓 포트와 겹치지 않게)
MANAGER_AUTHKEY = b'temp_authkey' # 인증키 설정 (바이트 문자열)

# 공유 데이터를 담을 딕셔너리
shared_tensor_data = {}

# 공유 데이터를 반환하는 함수
def get_shared_tensors():
    return shared_tensor_data

# 커스텀 매니저 클래스
class TensorManager(BaseManager):
    pass

# 매니저에 공유 데이터 접근 함수 등록
# 'get_tensors' 라는 이름으로 함수를 등록하고, DictProxy를 사용하도록 지정합니다.
TensorManager.register('get_tensors', callable=get_shared_tensors, proxytype=DictProxy)


def main():
    # 중요: 멀티프로세싱 시작 방법 설정 (fork가 CUDA와 더 잘 동작하는 경향이 있음)
    # 시스템 기본값이 spawn일 수 있으므로 명시적으로 설정하는 것이 좋을 수 있습니다.
    try:
        mp.set_start_method('fork', force=True)
        logging.info("Set multiprocessing start method to 'fork'.")
    except RuntimeError:
        logging.warning("Could not set start method to 'fork'. Using default.")


    logging.info("Loading model directly in the main process...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        logging.info(f"Starting model loading: {model_name}")
        load_start_time = time.time()
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
        model.eval()
        load_end_time = time.time()
        logging.info(f"Model '{model_name}' loaded to CUDA in {load_end_time - load_start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    logging.info("Sharing model parameters via IPC (using share_memory_)...")
    shared_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_cuda:
                try:
                    # 텐서를 공유 메모리로 이동 (내부적으로 IPC 준비)
                    param.share_memory_()
                    # 공유 딕셔너리에 텐서 자체를 저장
                    shared_tensor_data[name] = param
                    logging.info(f"Shared tensor prepared: {name} (Size: {param.size()}, Norm: {param.norm().item():.4f}, Device: {param.device})")
                    shared_count += 1
                except Exception as e:
                    logging.error(f"Failed to share tensor {name}: {e}")
            # else: Non-CUDA 파라미터는 무시

    logging.info(f"Prepared {shared_count} tensors for sharing.")
    if not shared_tensor_data:
        logging.warning("No tensors were prepared for sharing. Exiting.")
        return

    logging.info(f"Starting TensorManager server at {MANAGER_HOST}:{MANAGER_PORT}...")
    manager = TensorManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=MANAGER_AUTHKEY)

    try:
        # 매니저 서버 시작
        server = manager.get_server()
        logging.info("Manager server running. Waiting for client connections...")
        logging.info("Press Ctrl+C to stop.")
        server.serve_forever() # 블로킹 호출

    except OSError as bind_e:
        logging.error(f"Failed to bind manager server: {bind_e}. Port {MANAGER_PORT} might be in use.")
    except KeyboardInterrupt:
        logging.info("Received Ctrl+C. Shutting down manager server...")
    except Exception as e:
        logging.exception(f"An unexpected error occurred in manager server loop: {e}") # exception()은 스택 트레이스 포함
    finally:
        logging.info("Server shutting down.")


if __name__ == "__main__":
    main()
