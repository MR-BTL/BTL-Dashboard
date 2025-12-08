# 🚀 Streamlit Community Cloud Deployment Guide

This guide will help you deploy your Streamlit dashboard to Streamlit Community Cloud (free hosting).

## Prerequisites

1. ✅ Your code is pushed to a GitHub repository
2. ✅ Your repository is public (required for free tier)
3. ✅ Your `app.py` file is in the root directory
4. ✅ Your `requirements.txt` file is in the root directory

## Step-by-Step Deployment

### 1. Push Your Code to GitHub

If you haven't already, make sure your code is committed and pushed to GitHub:

```bash
git add .
git commit -m "Prepare for Streamlit Community Cloud deployment"
git push origin main
```

### 2. Deploy to Streamlit Community Cloud

1. **Go to Streamlit Community Cloud**
   - Visit: https://share.streamlit.io
   - Or: https://streamlit.io/cloud

2. **Sign in with GitHub**
   - Click "Sign in" or "Get started"
   - Authorize Streamlit to access your GitHub account

3. **Create a New App**
   - Click "New app" button
   - You'll see a form to configure your app

4. **Configure Your App**
   - **Repository**: Select `MR-BTL/BTL-Dashboard` (or your repo name)
   - **Branch**: Select `main` (or your default branch)
   - **Main file path**: Enter `app.py`
   - **App URL** (optional): Choose a custom subdomain like `btl-dashboard` (will be `btl-dashboard.streamlit.app`)

5. **Deploy**
   - Click "Deploy" button
   - Wait for the deployment to complete (usually 1-2 minutes)

### 3. Access Your Live App

Once deployed, your app will be available at:
- `https://your-app-name.streamlit.app`
- Or the auto-generated URL provided by Streamlit

## Post-Deployment

### Updating Your App

Every time you push changes to your GitHub repository, Streamlit Community Cloud will automatically:
1. Detect the changes
2. Rebuild your app
3. Deploy the new version

You can also manually trigger a redeploy from the Streamlit Community Cloud dashboard.

### Monitoring

- Check the "Manage app" section in Streamlit Community Cloud for:
  - Deployment logs
  - App status
  - Usage statistics
  - Error logs

## Troubleshooting

### Common Issues

1. **Deployment Fails**
   - Check that all dependencies in `requirements.txt` are correct
   - Verify your `app.py` file has no syntax errors
   - Check the deployment logs in Streamlit Community Cloud

2. **App Crashes on Load**
   - Check the logs in Streamlit Community Cloud dashboard
   - Ensure Google Sheets URLs are accessible (public sharing enabled)
   - Verify all required Python packages are in `requirements.txt`

3. **Import Errors**
   - Make sure all imports are listed in `requirements.txt`
   - Check that package names are correct (case-sensitive)

### File Structure Requirements

Your repository should have this structure:
```
BTL-Dashboard/
├── app.py                 # Main Streamlit app (required)
├── requirements.txt       # Python dependencies (required)
├── .streamlit/
│   └── config.toml       # Streamlit configuration (optional)
├── logo.png              # Your logo (optional)
└── README.md             # Documentation (optional)
```

## Important Notes

- ⚠️ **Public Repository Required**: Free tier requires your GitHub repository to be public
- 🔒 **No Secrets**: Don't commit API keys or secrets. Use Streamlit secrets management if needed
- 📊 **Google Sheets**: Make sure your Google Sheets are shared with "Anyone with the link can view"
- 🚀 **Auto-Deploy**: Changes are automatically deployed when you push to GitHub

## Resources

- [Streamlit Community Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Troubleshooting Guide](https://docs.streamlit.io/streamlit-community-cloud/troubleshooting)

---

**Need Help?** Check the Streamlit Community Cloud dashboard logs or visit the [Streamlit Community Forum](https://discuss.streamlit.io/).

