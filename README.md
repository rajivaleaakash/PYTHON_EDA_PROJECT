# 🏥 Insurance Cross-Sell Prediction - EDA Project

[![visual1](eda_visuals\bar_age_group.png)]
[![visual2](dashboard\age_wise_premium.png)]
[![visual3](dashboard\channel_customer_volume.png)]

A comprehensive Exploratory Data Analysis (EDA) project for analyzing customer behavior and predicting cross-sell opportunities in the insurance domain. This project provides automated insights, visualizations, and actionable business intelligence for insurance sales optimization.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Contact](#contact)

## 🎯 Project Overview

This project performs in-depth exploratory data analysis on insurance customer data to identify patterns and opportunities for cross-selling vehicle insurance to existing health insurance customers. The analysis includes customer demographics, vehicle information, premium analysis, and response prediction.

### Business Problem
Understanding which customers are most likely to be interested in vehicle insurance based on their:
- Demographics (age, gender, region)
- Vehicle characteristics (age, damage history)
- Existing insurance status
- Sales channel effectiveness

## ✨ Features

### Automated Analysis
- 🔍 **Data Cleaning Pipeline**: Automated data preprocessing and quality checks
- 📊 **KPI Dashboard**: 20+ key performance indicators across revenue, conversion, and operations
- 📈 **Interactive Visualizations**: High-quality charts and graphs for all key metrics
- 📝 **Automated Report Generation**: Comprehensive text-based EDA reports

### Key Analyses
1. **Gender-wise Premium Analysis**: Average premium comparison across genders
2. **Age-wise Premium Analysis**: Premium trends across age groups (18-25, 26-35, 36-45, etc.)
3. **Gender Balance Check**: Dataset balance assessment
4. **Vehicle Age Analysis**: Premium variations by vehicle age
5. **Response Rate by Vehicle Damage**: Conversion patterns for damaged vs. undamaged vehicles
6. **Previous Insurance Impact**: Premium distribution analysis
7. **Sales Channel Performance**: Top 10 channel analysis with response rates, volume, and revenue

### Advanced Features
- **Variance Detection**: Identify high/low variance features
- **Correlation Analysis**: Feature correlation with response variable
- **Anomaly Detection**: Z-score based outlier identification
- **Custom Logging**: Comprehensive logging system for debugging

## 📁 Project Structure

```
EDA_PROJECT_AAKASH_RAJIVALE/
│
├── data/                           # Raw data files
│   └── [your_data_files.csv]
│
├── clean_merge_dataset/            # Cleaned and merged datasets
│   └── clean_master_data.csv
│
├── dashboard/                      # Generated visualization files
│   ├── gender_wise_premium.png
│   ├── age_wise_premium.png
│   ├── gender_balance.png
│   ├── vehicle_age_wise_premium.png
│   ├── response_by_vehicle_damage.png
│   ├── premium_by_previous_insurance.png
│   ├── sales_channel_performance.png
│   └── correlation_heatmap.png
│
├── eda_visuals/                    # Additional EDA visualizations
│   ├── dist_*.png
│   └── bar_*.png
│
├── eda_report/                     # Text-based analysis reports
│   └── eda_report.txt
│
├── logs/                           # Application logs
│   └── eda_insights_*.log
│
├── myenv/                          # Virtual environment (not in git)
│
├── eda_script.py                   # Main data cleaning script
├── eda_insights.py                 # Analysis and visualization script
├── logging_setup.py                # Logging configuration
├── analysis.ipynb                  # Jupyter notebook for exploration
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/insurance-crosssell-eda.git
cd insurance-crosssell-eda
```

2. **Create virtual environment**
```bash
python -m venv myenv
```

3. **Activate virtual environment**

*Windows:*
```bash
myenv\Scripts\activate
```

*Linux/Mac:*
```bash
source myenv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Verify installation**
```bash
python --version
pip list
```

## 💻 Usage

### Running the Complete Analysis

1. **Data Cleaning and Preparation**
```bash
python eda_script.py
```
This script will:
- Load and clean the raw data
- Handle missing values
- Create the master dataset
- Save cleaned data to `clean_merge_dataset/`

2. **Generate Insights and Visualizations**
```bash
python eda_insights.py
```
This script will:
- Generate all visualizations
- Create comprehensive EDA report
- Calculate KPIs
- Perform correlation analysis
- Save outputs to `dashboard/`, `eda_visuals/`, and `eda_report/`

3. **Interactive Analysis**
```bash
jupyter notebook analysis.ipynb
```

### Output Files

After running the scripts, you'll get:

**📊 Visualizations** (`dashboard/` folder):
- Gender-wise premium comparison
- Age-wise premium analysis
- Gender balance charts
- Vehicle age analysis
- Response rate by vehicle damage
- Premium distribution by insurance status
- Sales channel performance (4-panel chart)
- Correlation heatmap

**📝 Report** (`eda_report/eda_report.txt`):
- Complete KPI summary
- All specific insights with numbers
- Sales channel performance tables
- Variance and correlation analysis

**📈 Additional Charts** (`eda_visuals/`):
- Distribution plots for numeric variables
- Bar charts for categorical variables

## 🔍 Key Insights

### Sample Findings (Based on typical insurance data)

**Revenue Metrics:**
- Total customers analyzed
- Total premium revenue generated
- Average premium per customer
- Potential revenue from interested customers

**Conversion Insights:**
- Overall response rate
- Response rate by vehicle damage history
- Impact of previous insurance on conversion

**Customer Segmentation:**
- Age group with highest premiums
- Gender-wise premium differences
- Vehicle age impact on premium
- Regional distribution patterns

**Sales Channel Performance:**
- Best converting channels (highest response rate)
- Highest volume channels (most customers)
- Premium channels (highest average premium)
- Channel efficiency matrix

## 📊 Visualizations

### Sample Visualizations Generated:

1. **Gender-wise Average Premium**
   - Bar chart comparing male vs female premiums
   - Value labels with rupee symbols

2. **Age-wise Premium Analysis**
   - Multi-colored bar chart for age groups
   - Clear trend visualization

3. **Sales Channel Performance Matrix**
   - 4-panel comprehensive analysis:
     - Response rate by channel
     - Customer volume by channel
     - Average premium by channel
     - Performance scatter plot (bubble chart)

4. **Correlation Heatmap**
   - Feature correlation matrix
   - Color-coded for easy interpretation

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization

### Development Tools
- **Jupyter Notebook**: Interactive analysis
- **Git**: Version control
- **Virtual Environment**: Dependency isolation

### Key Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📈 Key Performance Indicators (KPIs)

The project calculates and reports 10+ KPIs including:

**Revenue KPIs:**
- Total Premium Revenue
- Average Premium
- Median Premium
- Potential Revenue from Interested Customers

**Conversion KPIs:**
- Overall Response Rate
- Total Interested Customers
- Conversion Count

**Customer Segmentation KPIs:**
- Average Customer Age
- Previously Insured Rate
- New Customer Rate
- Vehicle Damage Rate
- Driving License Holders Rate

**Operational KPIs:**
- Total Sales Channels
- Most Active Channel
- Total Regions
- Total Customers

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Contact

**Aakash Rajivale**

- GitHub: [@https://github.com/rajivaleaakash]
- LinkedIn: [https://linkedin.com/in/rajivaleaakash]
- Email: rajivaleaakash@gmail.com

## 📚 Future Enhancements

- [ ] Machine learning model integration for prediction
- [ ] Interactive dashboard using Plotly/Dash
- [ ] Real-time data pipeline
- [ ] API endpoint for predictions
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Unit test coverage
- [ ] Performance optimization for large datasets

⭐ If you find this project helpful, please consider giving it a star!