#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  export_femx_public_snapshot.sh \
    --source /path/to/femx \
    --destination /path/to/femx-public \
    --branch publish/main-snapshot \
    --remote-url <git-url> \
    --commit-message "..."

Options:
  --source PATH          Source repository root (default: current script working directory)
  --destination PATH     Public destination path (default: ../femx-public from source root)
  --branch NAME          New branch name in destination repository
  --remote-name NAME     Remote name to configure/use for push (default: origin)
  --remote-url URL       Remote URL (used to set/overwrite remote name endpoint)
  --commit-message TEXT  Commit message
  --push                 Push branch after commit
  --help                 Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DEFAULT="${SCRIPT_DIR}/.."

SOURCE="${SRC_DEFAULT}"
DESTINATION="${SOURCE}/../femx-public"
BRANCH="publish/main-snapshot-$(date +%Y%m%d-%H%M%S)"
REMOTE_NAME="origin"
REMOTE_URL=""
COMMIT_MESSAGE=""
DO_PUSH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --destination)
      DESTINATION="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --remote-name)
      REMOTE_NAME="$2"
      shift 2
      ;;
    --remote-url)
      REMOTE_URL="$2"
      shift 2
      ;;
    --commit-message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --push)
      DO_PUSH=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${COMMIT_MESSAGE}" ]]; then
  SOURCE_COMMIT="unknown"
  if git -C "${SOURCE}" rev-parse --short HEAD >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git -C "${SOURCE}" rev-parse --short HEAD)"
  fi
  COMMIT_MESSAGE="chore(public): export femx public snapshot from ${SOURCE_COMMIT}"
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[error] rsync is required but not installed."
  exit 1
fi

if [[ ! -d "${SOURCE}" ]]; then
  echo "[error] Source path does not exist: ${SOURCE}"
  exit 1
fi

mkdir -p "${DESTINATION}"

if ! git -C "${SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] Source is not a git repository: ${SOURCE}"
  exit 1
fi

echo "[info] Source: ${SOURCE}"
echo "[info] Destination: ${DESTINATION}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
  echo "[info] destination is not a git repo. initializing..."
  git -C "${DESTINATION}" init
fi

if ! git -C "${DESTINATION}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] Destination is not a git repository: ${DESTINATION}"
  exit 1
fi

if git -C "${DESTINATION}" status --short --untracked-files=no | grep -q .; then
  echo "[error] Destination has uncommitted tracked changes. Commit or stash first."
  exit 1
fi

if git -C "${DESTINATION}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "[info] Branch already exists. checking out ${BRANCH}"
  git -C "${DESTINATION}" switch "${BRANCH}"
else
  echo "[info] creating branch ${BRANCH}"
  git -C "${DESTINATION}" switch -c "${BRANCH}"
fi

if [[ -n "${REMOTE_URL}" ]]; then
  git -C "${DESTINATION}" remote remove "${REMOTE_NAME}" >/dev/null 2>&1 || true
  git -C "${DESTINATION}" remote add "${REMOTE_NAME}" "${REMOTE_URL}"
fi

EXCLUDES=(
  --exclude='.git'
  --exclude='**/.git/**'
  --exclude='**/__pycache__/**'
  --exclude='**/.pytest_cache/**'
  --exclude='**/.mypy_cache/**'
  --exclude='**/.ruff_cache/**'
  --exclude='**/.ipynb_checkpoints/**'
  --exclude='**/.venv/**'
  --exclude='**/venv/**'
  --exclude='**/dist/**'
  --exclude='**/build/**'
  --exclude='**/.egg-info/**'
  --exclude='**/*.pyc'
  --exclude='**/*.swp'
)

SKIP_TEST_FILES=(
  "tests/architecture/test_documentation.py"
  "tests/unit/test_source_checkout_harness.py"
  "tests/integration/test_elmer_material_library.py"
  "tests/integration/test_source_checkouts.py"
)

echo "[copy] copying directories..."
for dir in \
  "docs/assets" \
  "docs/physics" \
  ".devin" \
  ".github" \
  "examples" \
  "scripts" \
  "src" \
  "tests"; do
  if [[ -d "${SOURCE}/${dir}" ]]; then
    echo "[copy-dir] ${dir}"
    mkdir -p "${DESTINATION}/$(dirname "${dir}")"
    rsync -a --delete "${EXCLUDES[@]}" "${SOURCE}/${dir}/" "${DESTINATION}/${dir}/"
  else
    echo "[skip-dir] ${dir} (not found)"
  fi
done

echo "[copy] copying files..."
for file in \
  "docs/INTEROPERABILITY.md" \
  "docs/MATERIALS.md" \
  "docs/ROADMAP.md" \
  ".gitignore" \
  ".pre-commit-config.yaml" \
  "CHANGELOG.md" \
  "CONTRIBUTING.md" \
  "LICENSE" \
  "Makefile" \
  "pyproject.toml" \
  "uv.lock"; do
  if [[ -f "${SOURCE}/${file}" ]]; then
    echo "[copy-file] ${file}"
    mkdir -p "${DESTINATION}/$(dirname "${file}")"
    cp "${SOURCE}/${file}" "${DESTINATION}/${file}"
  else
    echo "[skip-file] ${file} (not found)"
  fi
done

if [[ -f "${SOURCE}/README_PUBLIC.md" ]]; then
  echo "[copy-readme] README_PUBLIC.md -> README.md"
  cp "${SOURCE}/README_PUBLIC.md" "${DESTINATION}/README.md"
fi

echo "[prune] removing non-public test fixtures from destination"
for path in "${SKIP_TEST_FILES[@]}"; do
  if [[ -e "${DESTINATION}/${path}" ]]; then
    echo "[prune-file] ${path}"
    rm -f "${DESTINATION}/${path}"
  else
    echo "[prune-skip] ${path} (not found)"
  fi
done

echo "[patch-ci] normalizing public workflow test selectors"
CI_WORKFLOW="${DESTINATION}/.github/workflows/ci.yml"
if [[ -f "${CI_WORKFLOW}" ]]; then
  sed -i \
    -e 's|run: uv run pytest --cov=femx --cov-branch --cov-report=term-missing|run: uv run pytest --cov=femx --cov-branch --cov-report=term-missing -m "not contract"|' \
    -e 's|run: uv run pytest -m "unit or architecture or contract"|run: uv run pytest -m "unit or architecture"|' \
    "${CI_WORKFLOW}"
else
  echo "[patch-ci] skipping: ${CI_WORKFLOW} not found"
fi

cd "${DESTINATION}"

echo "[git] staging changes..."
git add -A

if git diff --cached --quiet; then
  echo "[info] no changes to commit."
  exit 0
fi

git commit -m "${COMMIT_MESSAGE}"
echo "[git] committed: ${COMMIT_MESSAGE}"

if [[ "${DO_PUSH}" == true ]]; then
  echo "[git] pushing branch ${BRANCH} to ${REMOTE_NAME}"
  git push -u "${REMOTE_NAME}" "${BRANCH}"
fi

echo "[done] branch ${BRANCH} is ready in ${DESTINATION}"
