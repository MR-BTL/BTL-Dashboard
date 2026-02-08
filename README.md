# Activation Agents Performance Dashboard

A comprehensive Streamlit-based dashboard for monitoring and analyzing activation agents' performance, tracking tasks, validating data quality, and monitoring consumer interactions across multiple metrics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Support](#support)

## Overview

This dashboard provides real-time insights into activation agents' operations, including task completion rates, interaction metrics, stock utilization, and data quality validations. It integrates seamlessly with Google Sheets as a data source and offers interactive visualizations for comprehensive performance analysis.

## Features

### Core Functionality

- **Agent Performance Tracking**: Monitor individual and team performance metrics
- **Task Management**: Track DCC, ECC, QR, and BBOS task completion
- **Interaction Analytics**: Analyze consumer interactions by type, brand, demographics, and location
- **Stock Management**: Monitor stock utilization, returns, and validation against interactions
- **Data Quality Validation**: Automated cross-validation between tasks and interactions
- **Distance Analysis**: Track and analyze agent travel distances (in/out distance metrics)

### Dashboard Sections

1. **Overview Dashboard**: High-level KPIs, top performers, and quick insights
2. **Operations & Performance**: Detailed agent and supervisor performance metrics, shift analysis, and distance analytics
3. **Interactions & Brands**: Brand performance, consumer demographics, and interaction trends
4. **Stock & Inventory**: Stock utilization rates, alerts, and detailed inventory tracking

### Advanced Features

- **Comprehensive Filtering**: Filter by date range, agent, zone, area, channel, supervisor, task status, shift duration, place, interaction type, age range, and interaction ID
- **Data Export**: Download filtered data as CSV or Excel files
- **Consolidated Reporting**: Single-table summary combining key metrics from all sections
- **Performance Optimization**: Optimized for large datasets with efficient data processing and display limits
- **Interactive Visualizations**: Plotly charts with drill-down capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Sheets with proper sharing permissions ("Anyone with the link can view")
- Internet connection for Google Sheets access

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MR-BTL/BTL-Dashboard.git
   cd BTL-Dashboard
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```
   
   Or use the helper script:
   ```bash
   python run_dashboard.py
   ```

5. **Access the dashboard**:
   - Open your browser to `http://localhost:8501`
   - The dashboard will open automatically

## Configuration

### Google Sheets Setup

1. Ensure your Google Sheets workbook contains the required sheets (see [Data Requirements](#data-requirements))
2. Share the Google Sheet with "Anyone with the link can view"
3. Copy the shareable link from the browser address bar
4. Paste the URL into the dashboard's sidebar input field

### Customization

- **Brand Colors**: Update the `BRAND` dictionary in `app.py` to match your brand colors
- **Logo**: Place your logo file (`logo.png`, `logo.jpg`, or `logo.jpeg`) in the project root or `assets/` directory
- **Logo Size**: Adjust `LOGO_WIDTH` in `app.py` to resize the logo display

## Usage

### Basic Workflow

1. **Load Data**: Paste your Google Sheets URL in the sidebar
2. **Apply Filters**: Use the sidebar filters to narrow down your analysis
3. **Navigate Tabs**: Explore different sections using the tab navigation
4. **Export Data**: Use download buttons to export filtered data
5. **View Details**: Expand sections to see detailed tables and charts

### Filtering

All filters work together to provide a comprehensive view:
- **Date Range**: Filter data by specific date ranges
- **Agent Selection**: Focus on specific agents or teams
- **Geographic Filters**: Filter by zone, area, or specific places
- **Task Filters**: Filter by status or shift duration
- **Interaction Filters**: Filter by type, age range, or search by ID

### Performance Tips

- For large datasets (>10,000 rows), the dashboard automatically optimizes display
- Tables show up to 1,000 rows by default (use download for full data)
- Charts are optimized to show up to 5,000 data points
- Filters are applied efficiently to minimize processing time

## Data Requirements

Your Google Sheets workbook must contain the following sheets with specified columns:

### Required Sheets

1. **Users**
   - `user-id` or `user_id`: Unique user identifier
   - `username`: Agent username
   - `zone`: Geographic zone
   - `area`: Area within zone
   - `channel`: Channel assignment
   - `sv`: Supervisor name
   - `role`: User role

2. **Tasks**
   - `user-id` or `user_id`: Agent identifier
   - `task-date` or `task_date`: Task date
   - `place-name` or `place_name`: Location name
   - `place-code` or `place_code`: Location code
   - `status`: Task status (completed, pending, in progress)
   - `check-in-time` or `check_in_time`: Check-in timestamp
   - `check-out-time` or `check_out_time`: Check-out timestamp
   - `DCC`, `ECC`, `QR`, `BBOS`: Task type counts
   - `in-distance` or `in_distance`: Distance to location (optional)
   - `out-distance` or `out_distance`: Distance from location (optional)

3. **interactions**
   - `user-id` or `user_id`: Agent identifier
   - `user-name` or `user_name`: Agent name
   - `day` or `date`: Interaction date
   - `Consumer interactions` or `consumer_interactions`: Interaction type
   - `Gender`: Consumer gender
   - `Age Range`: Consumer age range
   - `Main Brand`: Brand name
   - `Pack Purchase`: Purchase details
   - `place-name` or `place_name`: Location name
   - `URL`: Photo URL (optional)

4. **Stocklog**
   - `agent-name` or `agent_name`: Agent identifier
   - `date`: Stock transaction date
   - Transaction columns: `issued`, `used`, `returned`, `balance`
   - Transaction type: `release`, `back`, `used` (or aliases)

### Optional Sheets

- **sv-tasks**: Supervisor task assignments
- **Login**: User login timestamps
- **main**: Brand master list
- **purchase**: Purchase records

### Key Metrics Explained

- **DCC (Direct Consumer Contact)**: Direct face-to-face interactions with consumers
- **ECC (Enhanced Consumer Contact)**: Additional items provided with purchases (e.g., lighter, cricket accessories)
- **QR**: QR code scans for digital engagement
- **BBOS (Branded Bundle of Stock)**: Pack purchases with additional branded items (identified by "+" in Pack Purchase field)

## Deployment

### Streamlit Community Cloud (Free Hosting)

1. **Prepare Your Repository**:
   - Ensure all code is committed and pushed to GitHub
   - Repository must be public for free tier
   - Verify `app.py` and `requirements.txt` are in the root directory

2. **Deploy**:
   - Visit [Streamlit Community Cloud](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Configure:
     - **Repository**: Select your repository
     - **Branch**: `main` (or your default branch)
     - **Main file path**: `app.py`
     - **App URL**: Choose a custom subdomain (optional)
   - Click "Deploy"

3. **Access Your App**:
   - Your app will be available at `https://your-app-name.streamlit.app`
   - Changes pushed to GitHub automatically trigger redeployment

### Important Deployment Notes

- ⚠️ **Public Repository Required**: Free tier requires public GitHub repository
- 🔒 **Security**: Never commit API keys or secrets. Use Streamlit secrets management if needed
- 📊 **Google Sheets**: Ensure sheets are shared with "Anyone with the link can view"
- 🚀 **Auto-Deploy**: Changes are automatically deployed when pushed to GitHub

### Troubleshooting Deployment

**Deployment Fails**:
- Check deployment logs in Streamlit Community Cloud dashboard
- Verify all dependencies in `requirements.txt` are correct
- Ensure `app.py` has no syntax errors
- Confirm repository is public

**App Crashes**:
- Check logs tab in Streamlit dashboard
- Verify Google Sheets URLs are accessible
- Ensure all required Python packages are in `requirements.txt`
- Check that data sheets contain required columns

## Project Structure

```
BTL-Dashboard/
├── app.py                    # Main dashboard application
├── requirements.txt          # Python dependencies
├── run_dashboard.py         # Helper script to run dashboard
├── README.md                # This documentation
├── .gitignore               # Git ignore rules
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .devcontainer/
│   └── devcontainer.json    # Development container configuration
└── logo.png                 # Company logo (optional)
```

## Technical Details

### Technology Stack

- **Streamlit**: Web framework for building interactive dashboards
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualizations
- **OpenPyXL**: Excel file reading and writing
- **Requests**: HTTP library for Google Sheets access
- **Pillow**: Image processing for logo display

### Performance Optimizations

- **Data Type Optimization**: Automatic conversion to efficient data types (categories, downcasted numerics)
- **Efficient Filtering**: Vectorized operations and optimized filter chaining
- **Display Limits**: Tables limited to 1,000 rows, charts to 5,000 points
- **Memory Management**: Optimized DataFrame operations to reduce memory usage
- **Caching**: Streamlit caching for data loading (10-minute TTL)

### Data Processing

- **Robust Column Detection**: Handles various column name formats and aliases
- **Date Parsing**: Flexible date format handling (multiple formats supported)
- **Data Validation**: Cross-validation between tasks and interactions
- **Error Handling**: Graceful handling of missing data and format variations

## Support

### Getting Help

- **Documentation**: Refer to this README for detailed information
- **Issues**: Report bugs or request features via GitHub Issues
- **Questions**: Contact the development team for assistance

### Contributing

This is an internal project. For contributions or modifications, please coordinate with the development team.

## License

This project is proprietary software for internal use only.

---

**Built for Activation Agents Performance Tracking**

*Last Updated: February 2026*
