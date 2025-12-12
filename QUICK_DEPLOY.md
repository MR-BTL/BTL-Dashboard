# 🚀 Quick Deployment Checklist

## ✅ Pre-Deployment Status

- ✅ Code is committed
- ✅ Remote is configured: `MR-BTL/BTL-Dashboard`
- ✅ All required files present:
  - ✅ `app.py` (main application)
  - ✅ `requirements.txt` (dependencies)
  - ✅ `.streamlit/config.toml` (configuration)
- ✅ Repository is on GitHub

## 🎯 Deploy to Streamlit Community Cloud (5 minutes)

### Step 1: Go to Streamlit Community Cloud
👉 **https://share.streamlit.io**

### Step 2: Sign In
- Click **"Sign in"** or **"Get started"**
- Sign in with your **GitHub account** (MR-BTL)
- Authorize Streamlit to access your repositories

### Step 3: Create New App
- Click the **"New app"** button
- Fill in the form:

```
Repository: MR-BTL/BTL-Dashboard
Branch: main
Main file path: app.py
App URL (optional): btl-dashboard
```

### Step 4: Deploy
- Click **"Deploy"** button
- Wait 1-2 minutes for deployment
- Your app will be live at: **https://btl-dashboard.streamlit.app**

## 📋 Important Notes

⚠️ **Repository must be PUBLIC** for free hosting
- Go to: https://github.com/MR-BTL/BTL-Dashboard/settings
- Scroll to "Danger Zone" → Change visibility to Public

✅ **Google Sheets Access**
- Make sure your Google Sheets are shared with "Anyone with the link can view"
- Users will paste the Google Sheets URL in the dashboard

## 🔄 Updating Your App

After deployment, any push to GitHub will automatically redeploy:
```bash
git add .
git commit -m "Update dashboard"
git push origin main
```

## 🆘 Troubleshooting

**Deployment fails?**
- Check logs in Streamlit Community Cloud dashboard
- Verify `requirements.txt` has all dependencies
- Ensure repository is public

**App crashes?**
- Check the logs tab in Streamlit dashboard
- Verify Google Sheets URLs are accessible

---

**Ready?** Go to https://share.streamlit.io and deploy! 🚀

