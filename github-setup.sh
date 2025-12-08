#!/bin/bash

echo "==================================="
echo "GitHub Setup for BTL Dashboard"
echo "==================================="
echo ""

# Get user input
read -p "Enter your GitHub username: " GITHUB_USER
read -p "Enter your GitHub email: " GITHUB_EMAIL

# Configure Git
git config user.name "$GITHUB_USER"
git config user.email "$GITHUB_EMAIL"

echo ""
echo "✅ Git configured with:"
echo "   Username: $GITHUB_USER"
echo "   Email: $GITHUB_EMAIL"
echo ""

# Update remote
git remote remove btl 2>/dev/null
git remote remove origin 2>/dev/null

read -p "Which repository to push to? (1: MR-BTL/BTL-Dashboard, 2: Your own): " REPO_CHOICE

if [ "$REPO_CHOICE" = "1" ]; then
    REPO_URL="https://github.com/MR-BTL/BTL-Dashboard.git"
    echo "Setting up remote for: MR-BTL/BTL-Dashboard"
else
    read -p "Enter your repository name (e.g., username/repo-name): " REPO_NAME
    REPO_URL="https://github.com/$REPO_NAME.git"
    echo "Setting up remote for: $REPO_URL"
fi

git remote add origin "$REPO_URL"

echo ""
echo "==================================="
echo "Next Steps:"
echo "==================================="
echo "1. Create a Personal Access Token:"
echo "   → Go to: https://github.com/settings/tokens"
echo "   → Click 'Generate new token (classic)'"
echo "   → Name: 'Dashboard Upload'"
echo "   → Select scope: ✅ repo (full control)"
echo "   → Click 'Generate token'"
echo "   → COPY THE TOKEN!"
echo ""
echo "2. Push your code:"
echo "   git push -u origin main"
echo ""
echo "   When prompted for password, paste your TOKEN (not your password)"
echo ""
echo "==================================="
