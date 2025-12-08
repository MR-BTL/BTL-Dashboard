# 📊 Activation Agents Performance & Data Quality Dashboard

A comprehensive Streamlit dashboard for analyzing activation agents' performance, tracking tasks, validating data quality, and monitoring interactions across multiple metrics (DCC, ECC, QR, BBOS).

## 🌟 Features

### Overview
- **Total & Active/Inactive Agents** tracking
- **Task Metrics**: DCC, ECC, QR, and BBOS totals
- **Top 20 Agents** performance chart
- **Places vs Tasks** analysis

### Performance Metrics
- Date, Agent, Shift, and Place tracking
- Check-in/Check-out time monitoring
- Shift duration categorization (< 1 hour, 1-8 hours, > 8 hours)
- Automatic shift calculation from check-in/out times

### Data Validation
- **DCC/ECC/QR/BBOS Validation**: Cross-check between Tasks and Interactions
- **Stock Validation**: Verify stock usage vs actual interactions
- **Attendance Validation**: Track agents who logged in but had no interactions
- Visual mismatch detection with heatmaps and charts

### Interactions Analytics
- Consumer interactions breakdown (DCC, ECC, QR, BBOS)
- Main vs Occasional brand analysis
- Gender and age range demographics
- Pack purchase tracking with ECC item parsing
- Place-based interaction analysis
- Photo gallery with AppSheet URL support

### Tasks Analytics
- Detailed tasks table with place information
- Distance problem detection (> 100m)
- Status and shift distribution
- Agent performance summaries

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Sheets with proper permissions ("Anyone with the link can view")

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MR-BTL/BTL-Dashboard.git
cd BTL-Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## 📋 Data Requirements

Your Google Sheets workbook should contain the following sheets:

1. **Users**: user-id, username, zone, area, role
2. **Login**: user-id, timestamp
3. **Tasks**: user-id, task-date, DCC, ECC, QR, BBOS, place-name, check-in, check-out, shift, status, in-distance, out-distance
4. **Stocklog**: agent-name, date, issued, used, returned, balance
5. **interactions**: user-id, user-name, day, date, Consumer interactions, Gender, Age Range, Main Brand, Pack Purchase, place-name, URL (for photos)
6. **main**: List of main brands
7. **purchase**: Purchase records

## 🔧 Configuration

### Google Sheets Setup
1. Share your Google Sheet with "Anyone with the link can view"
2. Copy the shareable link
3. Paste it into the dashboard input field

### Customization
- Update `BRAND_COLORS` in `app.py` to match your brand colors
- Modify `LOGO_PATHS` to point to your logo location
- Adjust `LOGO_WIDTH` for logo sizing

## 📊 Key Metrics Explained

- **DCC (Direct Consumer Contact)**: Direct interactions with consumers
- **ECC (Enhanced Consumer Contact)**: Additional items given with purchases (e.g., lighter, cricket)
- **QR**: QR code scans
- **BBOS (Branded Bundle of Stock)**: Pack purchases with additional items (identified by "+" in Pack Purchase field)

## 🎨 Features Highlights

### Filters
- Date range
- Agent name
- Zone & Area
- Place name/code
- Interaction ID search
- Shift duration categories

### Export
- Download validation reports as Excel
- Filtered data export
- Per-metric validation files

### Visual Analytics
- Interactive Plotly charts
- Color-coded validation status
- Heatmaps for trend analysis
- Time-series graphs
- Distribution pie charts

## 🌐 Deployment

### Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `MR-BTL/BTL-Dashboard`
6. Main file: `app.py`
7. Click "Deploy"

Your dashboard will be live at: `https://your-app-name.streamlit.app`

## 📁 Project Structure

```
BTL-Dashboard/
├── app.py                 # Main dashboard application
├── requirements.txt       # Python dependencies
├── logo.png              # Company logo (optional)
├── README.md             # This file
├── Dashboard-guide.pages # User guide
└── .gitignore           # Git ignore rules
```

## 🛠️ Tech Stack

- **Streamlit**: Web framework
- **Pandas**: Data processing
- **Plotly**: Interactive visualizations
- **OpenPyXL**: Excel file handling
- **Pillow**: Image processing
- **Requests**: HTTP requests for Google Sheets

## 📝 License

This project is proprietary software for internal use.

## 👥 Support

For issues or questions, please contact the development team or create an issue in the repository.

## 🔄 Updates

Check the repository for the latest updates and improvements.

---

**Built with ❤️ for Activation Agents Performance Tracking**


