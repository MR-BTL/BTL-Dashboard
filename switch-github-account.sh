#!/bin/bash

echo "=========================================="
echo "Switch to Different GitHub Account"
echo "=========================================="
echo ""

# Clear existing Git credentials from macOS Keychain
echo "🔓 Clearing old GitHub credentials..."
git credential-osxkeychain erase << CRED
protocol=https
host=github.com
CRED

# Remove existing remotes
echo "🗑️  Removing old remotes..."
git remote remove origin 2>/dev/null
git remote remove btl 2>/dev/null

# Get new account information
echo ""
echo "📝 Enter your NEW GitHub account details:"
echo ""
read -p "GitHub Username: " NEW_USERNAME
read -p "GitHub Email: " NEW_EMAIL

# Configure Git with new account
git config user.name "$NEW_USERNAME"
git config user.email "$NEW_EMAIL"

echo ""
echo "✅ Git configured with:"
echo "   Username: $NEW_USERNAME"
echo "   Email: $NEW_EMAIL"
echo ""

# Ask which repository to use
echo "Choose repository option:"
echo "1) Push to MR-BTL/BTL-Dashboard (need collaborator access)"
echo "2) Create/use your own repository"
echo ""
read -p "Enter choice (1 or 2): " REPO_CHOICE

if [ "$REPO_CHOICE" = "1" ]; then
    REPO_URL="https://github.com/MR-BTL/BTL-Dashboard.git"
    echo ""
    echo "⚠️  Note: You need collaborator access to MR-BTL/BTL-Dashboard"
    echo "   Ask the owner to add: $NEW_USERNAME"
elif [ "$REPO_CHOICE" = "2" ]; then
    read -p "Enter your repository name (e.g., $NEW_USERNAME/dashboard-app): " REPO_NAME
    REPO_URL="https://github.com/$REPO_NAME.git"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Add new remote
git remote add origin "$REPO_URL"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Create a Personal Access Token:"
echo "   🔗 https://github.com/settings/tokens"
echo "   - Click 'Generate new token (classic)'"
echo "   - Name: Dashboard Access"
echo "   - Scope: ✅ repo (all)"
echo "   - Click 'Generate' and COPY the token"
echo ""
echo "2. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "   When prompted:"
echo "   Username: $NEW_USERNAME"
echo "   Password: [PASTE YOUR TOKEN HERE]"
echo ""
echo "3. Deploy on Streamlit Cloud:"
echo "   🔗 https://share.streamlit.io"
echo "   - Sign in with GitHub ($NEW_USERNAME)"
echo "   - Select repository: $REPO_URL"
echo "   - Main file: app.py"
echo "   - Click Deploy!"
echo ""
echo "=========================================="
