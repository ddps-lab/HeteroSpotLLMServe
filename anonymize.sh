#!/bin/bash
set -euo pipefail

###############################################################################
# anonymize.sh — 논문 아티팩트 제출용 익명화 ZIP 생성 스크립트
#
# 목적:
#   ShuntServe 저장소에서 git 이력과 GitHub 계정 정보(ddps-lab)를
#   모두 제거한 익명화된 ZIP 파일을 만든다.
#   단, ZIP 내에서도 `pip install -e .`로 vLLM을 설치할 수 있도록
#   git 메타데이터를 제거하기 *전에* 버전 정보와 wheel URL을 추출하여
#   install.sh 스크립트에 하드코딩해둔다.
#
# 전체 흐름:
#   1. vLLM 서브모듈을 main 브랜치로 전환
#   2. git describe로 vLLM 버전 문자열 추출 (PEP 440 형식 변환)
#   3. upstream vllm-project/vllm과의 merge-base를 구해 사전 빌드된 wheel URL 확인
#   4. 위 정보를 담은 install.sh 생성
#   5. 저장소를 임시 디렉토리에 복사 후 .git, .gitmodules, .gitignore 제거
#   6. ZIP으로 압축
###############################################################################

# ── 경로 설정 ──
REPO_DIR="./ShuntServe"          # 익명화할 저장소 경로 (스크립트를 저장소 상위에서 실행)
VLLM_DIR="$REPO_DIR/submodules/vLLM"     # vLLM 서브모듈 경로
ARTIFACT_NAME="ShuntServe-artifact"       # ZIP 내부 최상위 디렉토리 이름 (논문 제출용 이름)
OUTPUT_ZIP="${ARTIFACT_NAME}.zip"          # 생성될 ZIP 파일 이름

# ── 사전 조건 검증 ──
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: Repository directory $REPO_DIR not found."
    exit 1
fi

if [ ! -d "$VLLM_DIR" ] && [ ! -f "$VLLM_DIR/.git" ]; then
    echo "ERROR: vLLM submodule not found at $VLLM_DIR"
    exit 1
fi

# ── Step 1: vLLM 서브모듈을 main 브랜치로 전환 ──
# 서브모듈이 detached HEAD 상태일 수 있으므로, main 브랜치로 전환하여
# git describe가 정확한 태그 기반 버전을 반환하도록 한다.
echo "=== Step 1: Ensure vLLM submodule is on main branch ==="
pushd "$VLLM_DIR" > /dev/null

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "  Switching from '$CURRENT_BRANCH' to 'main'..."
    git switch main
fi
echo "  Branch: $(git branch --show-current)"

# ── Step 2: git 태그 기반 vLLM 버전 추출 ──
# git describe 출력 예시: v0.8.1-29-gc9cbe4b80
#   → "v0.8.1" 태그 이후 29개 커밋, 현재 커밋 해시 c9cbe4b80
# PEP 440 형식으로 변환: 0.8.1.dev29+gc9cbe4b80
#   → SETUPTOOLS_SCM_PRETEND_VERSION에 전달하면 .git 없이도 버전 인식 가능
echo "=== Step 2: Capture vLLM version from git tags ==="
# git describe output: v0.8.1-29-gc9cbe4b80
# PEP 440 conversion:  0.8.1.dev29+gc9cbe4b80
GIT_DESCRIBE=$(git describe --tags)
echo "  git describe: $GIT_DESCRIBE"

# sed 1차: "v" 접두사 제거 (v0.8.1 → 0.8.1)
# sed 2차: "-29-gc9cbe4b80" → ".dev29+gc9cbe4b80" (PEP 440 dev 릴리스 형식)
VLLM_VERSION=$(echo "$GIT_DESCRIBE" \
    | sed 's/^v//' \
    | sed 's/-\([0-9]\+\)-g/.dev\1+g/')
echo "  SETUPTOOLS_SCM_PRETEND_VERSION=$VLLM_VERSION"

# ── Step 3: 사전 빌드된 vLLM wheel URL 결정 ──
# vLLM의 setup.py는 빌드 시간을 줄이기 위해 사전 빌드된 wheel을 다운로드한다.
# wheel URL은 upstream vllm-project/vllm main과의 merge-base 커밋 해시로 결정된다.
# 1) GitHub API로 upstream main의 최신 커밋 SHA를 가져온다
# 2) 해당 커밋이 로컬에 없으면 fetch한다
# 3) upstream main과 우리 main의 merge-base를 구한다
# 4) 해당 커밋의 wheel URL이 존재하는지 HTTP 200으로 검증한다
# 5) 없으면 nightly wheel로 폴백한다
echo "=== Step 3: Determine precompiled wheel URL ==="
# Replicate the logic from setup.py:get_base_commit_in_main_branch()
# 1) Get upstream vllm-project/vllm main HEAD via GitHub API
# 2) Ensure the commit is available locally (fetch if needed)
# 3) Compute merge-base between upstream main and our branch
WHEEL_URL=""
BASE_COMMIT=""

UPSTREAM_JSON=$(curl -sf "https://api.github.com/repos/vllm-project/vllm/commits/main" 2>/dev/null || true)
if [ -n "$UPSTREAM_JSON" ]; then
    UPSTREAM_MAIN=$(echo "$UPSTREAM_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])" 2>/dev/null || true)
fi

if [ -n "${UPSTREAM_MAIN:-}" ]; then
    # upstream main 커밋이 로컬에 없으면 fetch
    if ! git cat-file -e "$UPSTREAM_MAIN" 2>/dev/null; then
        echo "  Fetching upstream main from vllm-project..."
        git fetch https://github.com/vllm-project/vllm main 2>/dev/null || true
    fi

    # upstream main과 우리 fork main의 공통 조상 (merge-base) 계산
    BASE_COMMIT=$(git merge-base "$UPSTREAM_MAIN" main 2>/dev/null || true)
fi

if [ -n "$BASE_COMMIT" ]; then
    # wheels.vllm.ai에서 해당 커밋의 사전 빌드 wheel이 있는지 확인
    CANDIDATE_URL="https://wheels.vllm.ai/${BASE_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$CANDIDATE_URL" 2>/dev/null || echo "000")
    if [ "$HTTP_STATUS" = "200" ]; then
        WHEEL_URL="$CANDIDATE_URL"
        echo "  Base commit: $BASE_COMMIT"
        echo "  Wheel URL verified (HTTP 200): $WHEEL_URL"
    else
        echo "  WARNING: Wheel for base commit returned HTTP $HTTP_STATUS, falling back to nightly."
    fi
fi

# merge-base wheel이 없으면 nightly 빌드로 폴백
if [ -z "$WHEEL_URL" ]; then
    WHEEL_URL="https://wheels.vllm.ai/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"
    echo "  Using nightly wheel URL: $WHEEL_URL"
fi

popd > /dev/null  # back to original directory

# ── Step 4: install.sh 생성 ──
# .git이 제거된 환경에서도 vLLM을 설치할 수 있도록,
# SETUPTOOLS_SCM_PRETEND_VERSION과 VLLM_PRECOMPILED_WHEEL_LOCATION을
# 하드코딩한 설치 스크립트를 저장소 내부에 생성한다.
echo "=== Step 4: Generate install.sh inside the repository ==="
cat > "$REPO_DIR/install.sh" << INSTALL_EOF
#!/bin/bash
set -euo pipefail

# install.sh — 이 아티팩트에서 수정된 vLLM을 설치하는 스크립트.
# .git 디렉토리가 제거되기 전에 버전 메타데이터가 캡처되었음.

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR/submodules/vLLM"

echo "Installing vLLM (editable mode with precompiled wheel)..."
SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" \\
VLLM_PRECOMPILED_WHEEL_LOCATION="${WHEEL_URL}" \\
pip install --editable .

export VLLM_USE_V1=0

echo ""
echo "Installation complete."
echo "Note: set VLLM_USE_V1=0 in your environment before running."
INSTALL_EOF

chmod +x "$REPO_DIR/install.sh"
echo "  Created $REPO_DIR/install.sh"

# ── Step 5: 임시 복사본 생성 후 git 메타데이터 제거 ──
# 원본 저장소를 건드리지 않기 위해 임시 디렉토리에 복사한 뒤,
# 모든 git 관련 파일을 삭제하여 익명화한다.
echo "=== Step 5: Create temporary copy and strip git metadata ==="
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT  # 스크립트 종료 시 임시 디렉토리 자동 삭제

# 저장소 전체를 임시 디렉토리에 복사 (아티팩트 이름으로 디렉토리명 변경)
cp -a "$REPO_DIR" "$TEMP_DIR/$ARTIFACT_NAME"

# .git 디렉토리/파일 제거 (메인 저장소 + 서브모듈의 .git 포함)
find "$TEMP_DIR/$ARTIFACT_NAME" -name ".git" -exec rm -rf {} + 2>/dev/null || true
# .gitmodules 제거 (ddps-lab GitHub URL이 포함되어 있어 익명화 위반)
rm -f "$TEMP_DIR/$ARTIFACT_NAME/.gitmodules"
# .gitignore 파일 제거 (필수는 아니지만 더 깔끔함)
find "$TEMP_DIR/$ARTIFACT_NAME" -name ".gitignore" -exec rm -f {} + 2>/dev/null || true

echo "  Stripped .git directories, .gitmodules, and .gitignore files"

# 검증: ddps-lab (연구실 GitHub 계정) 참조가 남아있는지 확인
DDPS_HITS=$(grep -r "github.com/ddps-lab" "$TEMP_DIR/$ARTIFACT_NAME" 2>/dev/null || true)
if [ -n "$DDPS_HITS" ]; then
    echo "  WARNING: Found remaining ddps-lab references:"
    echo "$DDPS_HITS"
else
    echo "  Verified: no github.com/ddps-lab references remain"
fi

# 검증: .git 디렉토리가 완전히 제거되었는지 확인
GIT_DIRS=$(find "$TEMP_DIR/$ARTIFACT_NAME" -name ".git" 2>/dev/null || true)
if [ -n "$GIT_DIRS" ]; then
    echo "  WARNING: Found remaining .git entries:"
    echo "$GIT_DIRS"
else
    echo "  Verified: no .git entries remain"
fi

# ── Step 6: ZIP 압축 ──
# 임시 디렉토리 내의 익명화된 저장소를 ZIP으로 압축한다.
echo "=== Step 6: Create ZIP archive ==="
rm -f "$OUTPUT_ZIP"
(cd "$TEMP_DIR" && zip -qr - "$ARTIFACT_NAME") > "$OUTPUT_ZIP"
ZIP_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
echo "  Created $OUTPUT_ZIP ($ZIP_SIZE)"

echo ""
echo "Done. To verify:"
echo "  1. unzip $OUTPUT_ZIP -d /tmp/verify"
echo "  2. cd /tmp/verify/$ARTIFACT_NAME"
echo "  3. bash install.sh"
