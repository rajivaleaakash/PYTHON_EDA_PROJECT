import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from logging_setup import setup_logger

logger = setup_logger("eda_insights")

DATA_PATH = Path("clean_merge_dataset/clean_master_data.csv")


def load_clean_master():
    """
    Load the cleaned master dataset generated from EDA pipeline.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        # Remove customer_id as it's not needed for analysis
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)
            logger.info("Removed customer_id from dataset")
        logger.info(f"Loaded clean master dataset: {DATA_PATH} | Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load clean master dataset: {e}")
        return pd.DataFrame()


# ---------------- VISUALIZATION ---------------- #

def plot_distribution(df, column, save=False):
    """Generate distribution plot for a numeric column."""
    try:
        plt.figure(figsize=(8,5))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)

        if save:
            path = Path("eda_visuals"); path.mkdir(exist_ok=True)
            file = path / f"dist_{column}.png"
            plt.savefig(file, dpi=300)
            logger.info(f"Saved distribution plot: {file}")

        plt.close()

    except Exception as e:
        logger.error(f"plot_distribution failed for {column}: {e}")


def plot_bar(df, column, save=False, top_n=None):
    """Create enhanced bar chart for categorical columns."""
    try:
        value_counts = df[column].value_counts()
        
        # Limit to top N if specified
        if top_n and len(value_counts) > top_n:
            value_counts = value_counts.head(top_n)
            title_suffix = f" (Top {top_n})"
        else:
            title_suffix = ""
        
        plt.figure(figsize=(max(10, len(value_counts) * 0.6), 6))
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(value_counts)))
        
        bars = plt.bar(range(len(value_counts)), value_counts.values, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Set x-axis labels
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        
        plt.title(f"Distribution: {column}{title_suffix}", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(column, fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / df.shape[0]) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()

        if save:
            path = Path("eda_visuals"); path.mkdir(exist_ok=True)
            file = path / f"bar_{column}.png"
            plt.savefig(file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved bar chart: {file}")

        plt.close()

    except Exception as e:
        logger.error(f"plot_bar failed for {column}: {e}")


# --------------- SPECIFIC INSIGHTS WITH GRAPHS --------------- #

def gender_wise_premium_analysis(df, save=False):
    """Analyze average annual premium by gender."""
    try:
        gender_premium = df.groupby('gender')['annual_premium [in Rs]'].mean()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        colors = ['#3498db', '#e74c3c']
        bars = plt.bar(gender_premium.index, gender_premium.values, color=colors, alpha=0.7, edgecolor='black')
        plt.title('Gender-wise Average Annual Premium', fontsize=14, fontweight='bold')
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Average Annual Premium (Rs)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if save:
            path = Path("dashboard"); path.mkdir(exist_ok=True)
            plt.savefig(path / "gender_wise_premium.png", dpi=300, bbox_inches='tight')
            logger.info("Saved gender-wise premium analysis")
        
        plt.close()
        return gender_premium
    
    except Exception as e:
        logger.error(f"gender_wise_premium_analysis failed: {e}")
        return pd.Series()


def age_wise_premium_analysis(df, save=False):
    """Analyze average annual premium by age groups."""
    try:
        # Create age bins
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        age_premium = df.groupby('age_group')['annual_premium [in Rs]'].mean()
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(age_premium)))
        bars = plt.bar(range(len(age_premium)), age_premium.values, color=colors, alpha=0.8, edgecolor='black')
        plt.xticks(range(len(age_premium)), age_premium.index, rotation=0)
        plt.title('Age-wise Average Annual Premium', fontsize=14, fontweight='bold')
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Average Annual Premium (Rs)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        if save:
            path = Path("dashboard"); path.mkdir(exist_ok=True)
            plt.savefig(path / "age_wise_premium.png", dpi=300, bbox_inches='tight')
            logger.info("Saved age-wise premium analysis")
        
        plt.close()
        return age_premium
    
    except Exception as e:
        logger.error(f"age_wise_premium_analysis failed: {e}")
        return pd.Series()


def gender_balance_analysis(df, save=False):
    """Check if data is balanced between genders."""
    try:
        gender_counts = df['gender'].value_counts()
        gender_pct = df['gender'].value_counts(normalize=True) * 100
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        colors = ['#3498db', '#e74c3c']
        bars1 = ax1.bar(gender_counts.index, gender_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Gender Distribution (Count)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Gender', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Pie chart
        ax2.pie(gender_pct.values, labels=gender_pct.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0.05))
        ax2.set_title('Gender Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = Path("dashboard"); path.mkdir(exist_ok=True)
            plt.savefig(path / "gender_balance.png", dpi=300, bbox_inches='tight')
            logger.info("Saved gender balance analysis")
        
        plt.close()
        return gender_counts, gender_pct
    
    except Exception as e:
        logger.error(f"gender_balance_analysis failed: {e}")
        return pd.Series(), pd.Series()


def vehicle_age_wise_premium_analysis(df, save=False):
    """Analyze average annual premium by vehicle age."""
    try:
        vehicle_premium = df.groupby('vehicle_age')['annual_premium [in Rs]'].mean().sort_values(ascending=False)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        colors = ['#27ae60', '#f39c12', '#e67e22']
        bars = plt.bar(range(len(vehicle_premium)), vehicle_premium.values, 
                       color=colors[:len(vehicle_premium)], alpha=0.8, edgecolor='black')
        plt.xticks(range(len(vehicle_premium)), vehicle_premium.index, rotation=0)
        plt.title('Vehicle Age-wise Average Annual Premium', fontsize=14, fontweight='bold')
        plt.xlabel('Vehicle Age', fontsize=12)
        plt.ylabel('Average Annual Premium (Rs)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if save:
            path = Path("dashboard"); path.mkdir(exist_ok=True)
            plt.savefig(path / "vehicle_age_wise_premium.png", dpi=300, bbox_inches='tight')
            logger.info("Saved vehicle age-wise premium analysis")
        
        plt.close()
        return vehicle_premium
    
    except Exception as e:
        logger.error(f"vehicle_age_wise_premium_analysis failed: {e}")
        return pd.Series()


def response_rate_by_vehicle_damage(df, save=False):
    """Additional Insight 1: Response rate by vehicle damage history."""
    try:
        damage_response = df.groupby('vehicle_damage')['response'].agg(['mean', 'count'])
        damage_response['response_rate'] = damage_response['mean'] * 100
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Response rate
        colors = ['#2ecc71', '#e74c3c']
        bars1 = ax1.bar(damage_response.index, damage_response['response_rate'].values,
                        color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Response Rate by Vehicle Damage History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Vehicle Damage', fontsize=12)
        ax1.set_ylabel('Response Rate (%)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Count distribution
        bars2 = ax2.bar(damage_response.index, damage_response['count'].values,
                        color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Customer Count by Vehicle Damage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Vehicle Damage', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = Path("dashboard"); path.mkdir(exist_ok=True)
            plt.savefig(path / "response_by_vehicle_damage.png", dpi=300, bbox_inches='tight')
            logger.info("Saved response rate by vehicle damage analysis")
        
        plt.close()
        return damage_response
    
    except Exception as e:
        logger.error(f"response_rate_by_vehicle_damage failed: {e}")
        return pd.DataFrame()


def premium_distribution_by_previous_insurance(df, save=False):
    """Additional Insight 2: Premium distribution by previous insurance status."""
    try:
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Box plot
        sns.boxplot(data=df, x='previously_insured', y='annual_premium [in Rs]',hue='previously_insured', palette=['#3498db', '#e74c3c'], legend=False)
        plt.title('Premium Distribution by Previous Insurance Status', fontsize=14, fontweight='bold')
        plt.xlabel('Previously Insured (0=No, 1=Yes)', fontsize=12)
        plt.ylabel('Annual Premium (Rs)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add mean values as text
        for i in [0, 1]:
            mean_val = df[df['previously_insured'] == i]['annual_premium [in Rs]'].mean()
            median_val = df[df['previously_insured'] == i]['annual_premium [in Rs]'].median()
            plt.text(i, mean_val, f'Mean: ₹{mean_val:,.0f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        if save:
            path = Path("dashboard"); path.mkdir(exist_ok=True)
            plt.savefig(path / "premium_by_previous_insurance.png", dpi=300, bbox_inches='tight')
            logger.info("Saved premium by previous insurance analysis")
        
        plt.close()
        
        # Return summary statistics
        summary = df.groupby('previously_insured')['annual_premium [in Rs]'].agg(['mean', 'median', 'std'])
        return summary
    
    except Exception as e:
        logger.error(f"premium_distribution_by_previous_insurance failed: {e}")
        return pd.DataFrame()


def sales_channel_performance(df, save=False):
    """Additional Insight 3: Sales channel performance analysis."""
    try:
        # Get top 10 sales channels by volume
        top_channels = df['sales_channel_code'].value_counts().head(10).index
        df_top = df[df['sales_channel_code'].isin(top_channels)]
        
        # Calculate metrics by channel
        channel_metrics = df_top.groupby('sales_channel_code').agg({
            'response': ['mean', 'count'],
            'annual_premium [in Rs]': 'mean'
        }).round(4)
        
        channel_metrics.columns = ['response_rate', 'customer_count', 'avg_premium']
        channel_metrics = channel_metrics.sort_values('response_rate', ascending=False)
        
        path = Path("dashboard"); path.mkdir(exist_ok=True)
        # Create visualization
        
        # 1. Response rate by channel
        plt.figure(figsize=(10, 6))
        colors1 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(channel_metrics)))
        x = range(len(channel_metrics))
        bars1 = plt.bar(x, channel_metrics['response_rate'] * 100,
                        color=colors1, edgecolor='black')
        plt.title("Response Rate by Sales Channel (Top 10)", fontsize=13, fontweight='bold')
        plt.xlabel("Sales Channel Code")
        plt.ylabel("Response Rate (%)")
        plt.xticks(x, channel_metrics.index, rotation=45)
        plt.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f"{height:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

        if save:
            plt.savefig(path / "channel_response_rate.png", dpi=300, bbox_inches='tight')
        plt.close()


        # 2. Customer count by channel
        channel_count = channel_metrics.sort_values('customer_count', ascending=False)
        plt.figure(figsize=(10, 6))
        colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(channel_count)))
        x = range(len(channel_count))
        bars2 = plt.bar(x, channel_count['customer_count'],
                        color=colors2, edgecolor='black')
        plt.title("Customer Volume by Sales Channel (Top 10)", fontsize=13, fontweight='bold')
        plt.xlabel("Sales Channel Code")
        plt.ylabel("Customer Count")
        plt.xticks(x, channel_count.index, rotation=45)
        plt.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f"{int(height):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        if save:
            plt.savefig(path / "channel_customer_volume.png", dpi=300, bbox_inches='tight')
        plt.close()


        # 3. Average premium by channel
        channel_premium = channel_metrics.sort_values('avg_premium', ascending=False)
        plt.figure(figsize=(10, 6))
        colors3 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(channel_premium)))
        x = range(len(channel_premium))
        bars3 = plt.bar(x, channel_premium['avg_premium'],
                        color=colors3, edgecolor='black')
        plt.title("Average Premium by Sales Channel (Top 10)", fontsize=13, fontweight='bold')
        plt.xlabel("Sales Channel Code")
        plt.ylabel("Average Premium (Rs)")
        plt.xticks(x, channel_premium.index, rotation=45)
        plt.grid(axis='y', alpha=0.3)

        for bar in bars3:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f"₹{height:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        if save:
            plt.savefig(path / "channel_avg_premium.png", dpi=300, bbox_inches='tight')
        plt.close()


        # 4. Scatter: Response Rate vs Customer Volume
        plt.figure(figsize=(10, 7))
        plt.scatter(channel_metrics['customer_count'],
                    channel_metrics['response_rate'] * 100,
                    s=channel_metrics['avg_premium'] / 50,
                    c=np.linspace(0.3, 0.9, len(channel_metrics)),
                    cmap='viridis', alpha=0.7, edgecolors='black')

        for idx, row in channel_metrics.iterrows():
            plt.annotate(str(idx),
                         (row['customer_count'], row['response_rate'] * 100),
                         fontsize=9, fontweight='bold')

        plt.title("Channel Performance Matrix (Bubble = Avg Premium)", fontsize=13, fontweight='bold')
        plt.xlabel("Customer Count")
        plt.ylabel("Response Rate (%)")
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(path / "channel_performance_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved all sales channel performance plots separately")
        return channel_metrics
    
    except Exception as e:
        logger.error(f"sales_channel_performance failed: {e}")
        return pd.DataFrame()


# --------------- AUTOMATED INSIGHTS --------------- #

def detect_variance(df, threshold=0.6):
    """Detect numeric columns with high or low variance."""
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns
        variances = {col: df[col].var() for col in numeric_cols}
        series = pd.Series(variances).sort_values(ascending=False)
        logger.info("Variance calculation completed.")
        return series
    except Exception as e:
        logger.error(f"detect_variance failed: {e}")
        return pd.Series()


def correlation_summary(df, target=None):
    """Return correlation summary with optional target variable."""
    try:
        corr = df.corr(numeric_only=True)
        if target:
            return corr[target].sort_values(ascending=False)
        return corr
    except Exception as e:
        logger.error(f"correlation_summary failed: {e}")
        return pd.DataFrame()


def calculate_kpis(df):
    """Calculate high-level KPIs from the cleaned dataset."""
    try:
        kpis = {}

        # Basic Revenue KPIs
        if "annual_premium [in Rs]" in df.columns:
            kpis["Total Customers"] = len(df)
            kpis["Avg Premium (Rs)"] = df["annual_premium [in Rs]"].mean()
            kpis["Median Premium (Rs)"] = df["annual_premium [in Rs]"].median()
            kpis["Total Premium Revenue (Rs)"] = df["annual_premium [in Rs]"].sum()

        # Conversion KPIs
        if "response" in df.columns:
            kpis["Overall Response Rate (%)"] = df["response"].mean() * 100
            kpis["Total Interested Customers"] = int(df["response"].sum())
            
            # Potential Revenue from Interested Customers
        if "annual_premium [in Rs]" in df.columns:
            interested_customers = df[df['response'] == 1]
            if len(interested_customers) > 0:
                kpis["Avg Premium - Interested (Rs)"] = interested_customers["annual_premium [in Rs]"].mean()
                kpis["Potential Revenue - Interested (Rs)"] = interested_customers["annual_premium [in Rs]"].sum()

        # Customer Segmentation KPIs
        if "previously_insured" in df.columns:
            kpis["Previously Insured Rate (%)"] = df["previously_insured"].mean() * 100
            kpis["New Customer Rate (%)"] = (1 - df["previously_insured"].mean()) * 100

        if "age" in df.columns:
            kpis["Avg Customer Age"] = df["age"].mean()
            kpis["Median Customer Age"] = df["age"].median()
        
        if "vintage" in df.columns:
            kpis["Avg Customer Vintage (days)"] = df["vintage"].mean()
        
        if "vehicle_damage" in df.columns:
            kpis["Vehicle Damage Rate (%)"] = (df["vehicle_damage"] == "Yes").mean() * 100
        
        if "driving_licence_present" in df.columns:
            kpis["Driving License Holders (%)"] = df["driving_licence_present"].mean() * 100
        
        # Operational KPIs
        if "sales_channel_code" in df.columns:
            kpis["Total Sales Channels"] = df["sales_channel_code"].nunique()
            kpis["Most Active Channel"] = int(df["sales_channel_code"].mode()[0]) if len(df["sales_channel_code"].mode()) > 0 else "N/A"
        
        if "region_code" in df.columns:
            kpis["Total Regions"] = df["region_code"].nunique()

        logger.info("KPI calculations complete.")
        return kpis

    except Exception as e:
        logger.error(f"calculate_kpis failed: {e}")
        return {}


def detect_anomalies(df, date_col, value_col, z_threshold=3):
    """Detect anomalies in a time-series column using Z-score."""
    try:
        df = df.sort_values(date_col)
        df["z_score"] = (df[value_col] - df[value_col].mean()) / df[value_col].std()
        anomalies = df[df["z_score"].abs() > z_threshold]
        logger.info(f"Anomalies detected: {len(anomalies)} records")
        return anomalies
    except Exception as e:
        logger.error(f"detect_anomalies failed: {e}")
        return pd.DataFrame()


# --------------- EDA REPORT GENERATOR --------------- #

def generate_eda_report(df, name="eda_report"):
    """
    Create automated EDA report including:
    - KPIs
    - Specific insights
    - Sales channel performance
    - Variance summary
    - Correlation matrix
    """
    try:
        report_dir = Path("eda_report"); report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"{name}.txt"

        with open(report_path, "w", encoding='utf-8') as f:
            f.write("\n")
            f.write("-"*50+" AUTOMATED EDA REPORT "+"-"*50)
            f.write("\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
            f.write("*" * 80 + "\n\n")

            # KPIs Section
            f.write("-"*50+ " KEY PERFORMANCE INDICATORS" + "-"*50)
            f.write("\n")
            
            kpis = calculate_kpis(df)
            
            # Revenue KPIs
            f.write("📊 REVENUE METRICS:\n")
            f.write("-" * 50 + "\n")
            revenue_kpis = ["Total Customers", "Total Premium Revenue (Rs)", 
                           "Avg Premium (Rs)", "Median Premium (Rs)",
                           "Potential Revenue - Interested (Rs)", "Avg Premium - Interested (Rs)"]
            for key in revenue_kpis:
                if key in kpis:
                    if "Rs" in key and key != "Avg Premium (Rs)" and key != "Median Premium (Rs)" and key != "Avg Premium - Interested (Rs)":
                        f.write(f"   • {key:<45}: ₹{kpis[key]:,.2f}\n")
                    elif "Rs" in key:
                        f.write(f"   • {key:<45}: ₹{kpis[key]:,.2f}\n")
                    else:
                        f.write(f"   • {key:<45}: {kpis[key]:,.0f}\n")
            f.write("\n")
            
            # Conversion KPIs
            f.write("🎯 CONVERSION METRICS:\n")
            f.write("-" * 80 + "\n")
            conversion_kpis = ["Overall Response Rate (%)", "Total Interested Customers"]
            for key in conversion_kpis:
                if key in kpis:
                    if "%" in key:
                        f.write(f"   • {key:<45}: {kpis[key]:.2f}%\n")
                    else:
                        f.write(f"   • {key:<45}: {kpis[key]:,.0f}\n")
            f.write("\n")
            
            # Customer Segmentation KPIs
            f.write("👥 CUSTOMER SEGMENTATION:\n")
            f.write("-" * 80 + "\n")
            customer_kpis = ["Avg Customer Age", "Median Customer Age", 
                           "Previously Insured Rate (%)", "New Customer Rate (%)",
                           "Avg Customer Vintage (days)", "Vehicle Damage Rate (%)",
                           "Driving License Holders (%)"]
            for key in customer_kpis:
                if key in kpis:
                    if "%" in key:
                        f.write(f"   • {key:<45}: {kpis[key]:.2f}%\n")
                    else:
                        f.write(f"   • {key:<45}: {kpis[key]:,.2f}\n")
            f.write("\n")
            
            # Operational KPIs
            f.write("🏢 OPERATIONAL METRICS:\n")
            f.write("-" * 80 + "\n")
            operational_kpis = ["Total Sales Channels", "Most Active Channel", "Total Regions"]
            for key in operational_kpis:
                if key in kpis:
                    f.write(f"   • {key:<45}: {kpis[key]}\n")
            f.write("\n\n")

            # Specific Insights Section
            f.write("\n")
            f.write("-" * 50 + "SPECIFIC INSIGHTS" + "-" * 50 + "\n")
            f.write("\n")
            
            # i. Gender-wise average premium
            f.write("1. Gender-wise Average Annual Premium:\n")
            f.write("-" * 80 + "\n")
            gender_premium = gender_wise_premium_analysis(df, save=True)
            for gender, premium in gender_premium.items():
                f.write(f"   • {gender:<15}: ₹{premium:,.2f}\n")
            f.write("\n\n")
            
            # ii. Age-wise average premium
            f.write("2. Age-wise Average Annual Premium:\n")
            f.write("-" * 80 + "\n")
            age_premium = age_wise_premium_analysis(df, save=True)
            for age_group, premium in age_premium.items():
                f.write(f"   • {age_group:<15}: ₹{premium:,.2f}\n")
            f.write("\n\n")
            
            # iii. Gender balance
            f.write("3. Gender Balance Analysis:\n")
            f.write("-" * 80 + "\n")
            gender_counts, gender_pct = gender_balance_analysis(df, save=True)
            for gender in gender_counts.index:
                f.write(f"   • {gender:<15}: {gender_counts[gender]:,} ({gender_pct[gender]:.2f}%)\n")
            
            # Calculate balance ratio
            if len(gender_counts) == 2:
                ratio = min(gender_counts.values) / max(gender_counts.values)
                f.write(f"\n   Balance Ratio: {ratio:.2f} (1.0 = perfectly balanced)\n")
                if ratio > 0.9:
                    f.write("   ✓ Status: Data is well-balanced between genders\n")
                else:
                    f.write("   ⚠ Status: Data shows some imbalance between genders\n")
            f.write("\n\n")
            
            # iv. Vehicle age-wise average premium
            f.write("4. Vehicle Age-wise Average Annual Premium:\n")
            f.write("-" * 80 + "\n")
            vehicle_premium = vehicle_age_wise_premium_analysis(df, save=True)
            for vehicle_age, premium in vehicle_premium.items():
                f.write(f"   • {vehicle_age:<15}: ₹{premium:,.2f}\n")
            f.write("\n\n")
            
            # 1. Response Rate by Vehicle Damage
            f.write("5. Response Rate by Vehicle Damage History:\n")
            f.write("-" * 80 + "\n")
            damage_response = response_rate_by_vehicle_damage(df, save=True)
            for damage, row in damage_response.iterrows():
                f.write(f"   • {damage:<15}: {row['response_rate']:.2f}% (n={int(row['count']):,} customers)\n")
            f.write("\n\n")
            
            # 2. Premium Distribution by Previous Insurance
            f.write("6. Premium Distribution by Previous Insurance Status:\n")
            f.write("-" * 80 + "\n")
            prev_ins_summary = premium_distribution_by_previous_insurance(df, save=True)
            for status, row in prev_ins_summary.iterrows():
                status_label = "No" if status == 0 else "Yes"
                f.write(f"   Previously Insured ({status_label}):\n")
                f.write(f"      • Mean Premium        : ₹{row['mean']:,.2f}\n")
                f.write(f"      • Median Premium      : ₹{row['median']:,.2f}\n")
                f.write(f"      • Std Deviation       : ₹{row['std']:,.2f}\n")
            f.write("\n")
            
            # 3. Sales Channel Performance Analysis
            f.write("7. Sales Channel Performance Analysis (Top 10 Channels):\n")
            f.write("-" * 80 + "\n")
            channel_metrics = sales_channel_performance(df, save=True)
            
            if not channel_metrics.empty:
                f.write("\n Top 5 Channels by Response Rate:\n")
                f.write("   " + "-" * 76 + "\n")
                f.write(f"   {'Channel':<12} {'Response Rate':<18} {'Customers':<15} {'Avg Premium':<20}\n")
                f.write("   " + "-" * 76 + "\n")
                
                for idx, row in channel_metrics.head(5).iterrows():
                    f.write(f"   {str(idx):<12} {row['response_rate']*100:>8.2f}%        ")
                    f.write(f"{int(row['customer_count']):>10,}     ₹{row['avg_premium']:>12,.2f}\n")
                
                f.write("\n Top 5 Channels by Customer Volume:\n")
                f.write("   " + "-" * 76 + "\n")
                channel_volume = channel_metrics.sort_values('customer_count', ascending=False)
                f.write(f"   {'Channel':<12} {'Customers':<18} {'Response Rate':<15} {'Avg Premium':<20}\n")
                f.write("   " + "-" * 76 + "\n")
                
                for idx, row in channel_volume.head(5).iterrows():
                    f.write(f"   {str(idx):<12} {int(row['customer_count']):>10,}        ")
                    f.write(f"{row['response_rate']*100:>8.2f}%    ₹{row['avg_premium']:>12,.2f}\n")
                
                f.write("\n Top 5 Channels by Average Premium:\n")
                f.write("   " + "-" * 76 + "\n")
                channel_premium = channel_metrics.sort_values('avg_premium', ascending=False)
                f.write(f"   {'Channel':<12} {'Avg Premium':<20} {'Customers':<15} {'Response Rate':<15}\n")
                f.write("   " + "-" * 76 + "\n")
                
                for idx, row in channel_premium.head(5).iterrows():
                    f.write(f"   {str(idx):<12} ₹{row['avg_premium']:>12,.2f}     ")
                    f.write(f"{int(row['customer_count']):>10,}    {row['response_rate']*100:>8.2f}%\n")
                
                # Channel Performance Summary
                f.write("\n Channel Performance Summary:\n")
                f.write("   " + "-" * 76 + "\n")
                
                # Best converting channel
                best_convert = channel_metrics.index[0]
                best_convert_rate = channel_metrics.iloc[0]['response_rate'] * 100
                f.write(f"   • Best Converting Channel      : {best_convert} ({best_convert_rate:.2f}% response rate)\n")
                
                # Highest volume channel
                highest_vol = channel_volume.index[0]
                highest_vol_count = int(channel_volume.iloc[0]['customer_count'])
                f.write(f"   • Highest Volume Channel       : {highest_vol} ({highest_vol_count:,} customers)\n")
                
                # Highest premium channel
                highest_prem = channel_premium.index[0]
                highest_prem_val = channel_premium.iloc[0]['avg_premium']
                f.write(f"   • Highest Avg Premium Channel  : {highest_prem} (₹{highest_prem_val:,.2f})\n")
                
                # Overall channel statistics
                f.write(f"\n   • Total Active Channels (Top 10): {len(channel_metrics)}\n")
                f.write(f"   • Avg Response Rate Across Top 10: {channel_metrics['response_rate'].mean()*100:.2f}%\n")
                f.write(f"   • Total Customers in Top 10     : {int(channel_metrics['customer_count'].sum()):,}\n")
                f.write(f"   • Avg Premium Across Top 10     : ₹{channel_metrics['avg_premium'].mean():,.2f}\n")
            
            f.write("\n\n")

            # Variance Summary
            f.write("\n")
            f.write("-" * 50 + " VARIANCE SUMMARY " + "-" * 50 + "\n")
            f.write("\n")
            variance_series = detect_variance(df)
            f.write(str(variance_series))
            f.write("\n\n")

            # Correlation Summary
            f.write("\n")
            f.write("-" * 50 + " CORRELATION WITH RESPONSE " + "-" * 50 + "\n")
            f.write("\n")
            if 'response' in df.columns:
                corr_series = correlation_summary(df, target='response')
                f.write(str(corr_series))
            else:
                f.write("Response column not found in dataset\n")
            
            f.write("\n\n")
            f.write(" "*50+" END OF REPORT\n")
            

        logger.info(f"EDA Report generated: {report_path}")

    except Exception as e:
        logger.error(f"generate_eda_report failed: {e}")


# --------------- DASHBOARD --------------- #

def generate_dashboard(df):
    """Generate charts and correlation heatmap and save them as dashboard folder."""
    try:
        dash = Path("dashboard"); dash.mkdir(exist_ok=True)

        # Generate specific insights
        logger.info("Generating specific insights...")
        gender_wise_premium_analysis(df, save=True)
        age_wise_premium_analysis(df, save=True)
        gender_balance_analysis(df, save=True)
        vehicle_age_wise_premium_analysis(df, save=True)
        response_rate_by_vehicle_damage(df, save=True)
        premium_distribution_by_previous_insurance(df, save=True)
        sales_channel_performance(df, save=True)

        # Numeric distributions
        for col in df.select_dtypes(include=np.number).columns:
            plot_distribution(df, col, save=True)

        # Categorical
        for col in df.select_dtypes(include="category").columns:
            plot_bar(df, col, save=True, top_n=15)

        # Correlation heatmap
        try:
            plt.figure(figsize=(12, 10))
            numeric_df = df.select_dtypes(include=np.number)
            sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(dash / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        except:
            logger.warning("Could not generate heatmap.")

        logger.info("Dashboard generated successfully.")

    except Exception as e:
        logger.error(f"generate_dashboard failed: {e}")


# --------------- MAIN EXECUTION --------------- #

if __name__ == "__main__":
    df = load_clean_master()

    if df.empty:
        logger.error("No data to analyze.")
    else:
        logger.info("Starting EDA analysis...")
        generate_dashboard(df)
        generate_eda_report(df)
        logger.info("EDA Insights module completed.")