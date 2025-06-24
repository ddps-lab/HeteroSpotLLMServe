#!/bin/bash

# 동기화할 원격 서버 IP 주소 목록 (하드코딩)
REMOTE_IPS=(
    "172.31.46.87"
    #"172.31.46.87" 
)

SOURCE_DIR="/home/ubuntu/vllm-example"

# 원격 서버 사용자 이름
REMOTE_USER="ubuntu"

# 원격 서버의 대상 디렉토리 경로
REMOTE_BASE_DIR="/home/ubuntu/vllm-example"

# rsync 제외 패턴 목록
# 각 패턴은 '--exclude' 옵션으로 전달됩니다.
EXCLUDE_PATTERNS=(
    "__pycache__"
    ".git"          # Git 관련 디렉토리 제외
    "*.pyc"         # Python 컴파일 캐시 파일 제외
    "*.so"          # 컴파일된 공유 라이브러리 제외 (소스만 동기화하려는 경우)
    "*.o"           # 컴파일된 오브젝트 파일 제외
    ".ipynb_checkpoints" # Jupyter 노트북 체크포인트 제외
    "logs/"         # 로그 디렉토리 제외 (필요시)
    "data/"         # 데이터 디렉토리 제외 (필요시)
    "build/"        # 빌드 디렉토리 제외
    "dist/"         # 배포 디렉토리 제외
)

# 제외 옵션을 rsync 명령 형식으로 변환
EXCLUDE_ARGS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_ARGS+=("--exclude=$pattern")
done

echo "Workspace 동기화를 시작합니다..."
echo "로컬 경로: $SOURCE_DIR (현재 디렉토리)"
echo "원격 사용자: $REMOTE_USER"
echo "원격 기본 경로: $REMOTE_BASE_DIR"
echo "제외 패턴: ${EXCLUDE_PATTERNS[*]}"
echo "-------------------------------------"

# 각 IP 주소에 대해 rsync 실행
for ip in "${REMOTE_IPS[@]}"; do
    echo ">>> [$ip] 서버로 동기화 중..."

    # 원격 서버에 대상 디렉토리 생성 시도 (없을 경우 대비)
    # 비밀번호 없이 SSH 접속이 가능해야 함
    ssh "${REMOTE_USER}@${ip}" "mkdir -p ${REMOTE_BASE_DIR}"

    # rsync 명령어 실행
    rsync -avz --delete "${EXCLUDE_ARGS[@]}" "${SOURCE_DIR}/" "${REMOTE_USER}@${ip}:${REMOTE_BASE_DIR}/"

    # 동기화 결과 확인 (rsync 종료 코드로 확인)
    if [ $? -eq 0 ]; then
        echo ">>> [$ip] 동기화 성공."
    else
        echo ">>> [$ip] 동기화 실패. 오류를 확인하세요."
    fi
    echo "-------------------------------------"
done

echo "모든 동기화 작업 완료."
