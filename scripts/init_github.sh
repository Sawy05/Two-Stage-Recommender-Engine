#!/usr/bin/env bash
set -euo pipefail

REPO_SLUG="Sawy05/two-stage-recsys-dcn"
BRANCH="main"

echo "============================================================"
echo " Initializing GitHub Repository: ${REPO_SLUG}"
echo "============================================================"

if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is not installed or not on PATH." >&2
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "[ERROR] GitHub CLI ('gh') is not installed. Install from https://cli.github.com/" >&2
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "[ERROR] GitHub CLI is not authenticated. Run 'gh auth login' first." >&2
    exit 1
fi

if [ ! -d ".git" ]; then
    echo "[INFO] Initializing local Git repository on branch '${BRANCH}'..."
    git init -b "${BRANCH}"
else
    echo "[INFO] Existing Git repository detected. Ensuring branch '${BRANCH}'..."
    git checkout -B "${BRANCH}"
fi

echo "[INFO] Staging repository files..."
git add .

if git diff --cached --quiet; then
    echo "[INFO] No new changes to commit."
else
    git commit -m "feat: enterprise two-stage recommender with Two-Tower ANN, DCNv2 ranking, and LinUCB bandits"
fi

if gh repo view "${REPO_SLUG}" >/dev/null 2>&1; then
    echo "[INFO] Remote repository '${REPO_SLUG}' already exists on GitHub."
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "https://github.com/${REPO_SLUG}.git"
    fi
    git push -u origin "${BRANCH}"
else
    echo "[INFO] Creating public repository '${REPO_SLUG}' and pushing '${BRANCH}'..."
    gh repo create "${REPO_SLUG}" --public --source=. --remote=origin --push
fi

echo "[SUCCESS] Published to https://github.com/${REPO_SLUG}"
