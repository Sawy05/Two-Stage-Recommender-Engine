#!/usr/bin/env bash
set -euo pipefail

REPO_OWNER="Sawy05"
REPO_NAME="two-stage-recsys-dcn"
REPO_SLUG="${REPO_OWNER}/${REPO_NAME}"
BRANCH="main"
COMMIT_MSG="docs: publish comprehensive thesis, system proposal, and CI compilation workflow"

echo "================================================================================"
echo " Automated Documentation & Thesis Git Sync: ${REPO_SLUG}"
echo "================================================================================"

# 1. Validate Git presence
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] Git is not installed or not in system PATH." >&2
    exit 1
fi

# 2. Check and configure Git identity if unconfigured
if [ -z "$(git config user.name)" ]; then
    git config user.name "Sawy05"
fi
if [ -z "$(git config user.email)" ]; then
    git config user.email "Sawy05@users.noreply.github.com"
fi

# 3. Detect or configure remote origin
if git remote get-url origin >/dev/null 2>&1; then
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "[INFO] Existing remote origin detected: ${CURRENT_REMOTE}"
else
    echo "[INFO] Configuring remote origin to https://github.com/${REPO_SLUG}.git..."
    git remote add origin "https://github.com/${REPO_SLUG}.git"
fi

# 4. Check for GitHub CLI (gh) authentication (or warn if not present)
HAS_GH=false
if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
        HAS_GH=true
        echo "[INFO] GitHub CLI ('gh') is authenticated."
    else
        echo "[WARN] GitHub CLI ('gh') is installed but not authenticated. Run 'gh auth login' to enable repo auto-creation."
    fi
else
    echo "[WARN] GitHub CLI ('gh') is not installed. Pushes will proceed via standard Git remote credentials."
fi

# 5. Check if remote repo exists on GitHub if gh is active
if [ "$HAS_GH" = true ]; then
    if ! gh repo view "${REPO_SLUG}" >/dev/null 2>&1; then
        echo "[INFO] Creating public repository '${REPO_SLUG}' on GitHub..."
        gh repo create "${REPO_SLUG}" --public --source=. --remote=origin || true
    fi
fi

# 6. Ensure .gitignore excludes LaTeX build artifacts while tracking documentation
if ! grep -q "\*.aux" .gitignore; then
    echo "[INFO] Appending LaTeX build artifacts to .gitignore..."
    cat << 'EOF' >> .gitignore

# LaTeX build artifacts
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.synctex.gz
docs/report/latex/*.pdf
EOF
fi

# 7. Stage documentation and workflow files
echo "[INFO] Staging thesis, LaTeX reports, and GitHub workflows..."
git add docs/report/THESIS.md
git add docs/report/latex/
git add .github/workflows/compile-report.yml
git add scripts/publish_report.sh
git add .gitignore

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "[INFO] Working tree is clean. No new documentation changes to commit."
else
    echo "[INFO] Creating commit: '${COMMIT_MSG}'"
    git commit -m "${COMMIT_MSG}"
fi

# 8. Push to GitHub origin main
echo "[INFO] Pushing changes to origin ${BRANCH}..."
if git push -u origin "${BRANCH}"; then
    echo ""
    echo "================================================================================"
    echo " [SUCCESS] Thesis and technical reports published successfully!"
    echo " Repository URL : https://github.com/${REPO_SLUG}"
    echo " Markdown Thesis: https://github.com/${REPO_SLUG}/blob/${BRANCH}/docs/report/THESIS.md"
    echo " LaTeX Sources  : https://github.com/${REPO_SLUG}/tree/${BRANCH}/docs/report/latex"
    echo " CI Actions     : https://github.com/${REPO_SLUG}/actions"
    echo "================================================================================"
else
    echo "[WARN] Direct git push failed. If remote authentication is required, ensure your SSH key or Personal Access Token (PAT) is configured, or run 'gh auth login'."
    echo "[INFO] Local commit has been recorded successfully on branch '${BRANCH}'."
    echo "To push manually: git push -u origin ${BRANCH}"
fi
