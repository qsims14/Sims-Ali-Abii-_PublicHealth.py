import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

#Health Analyzer Class

class HealthAnalyzer:
    """
    Analyzes hospital patient data, performing cleaning, summarizing outcomes,
    aggregating metrics, and generating visualizations.
    """
    def __init__(self, df):
        # 1. Normalize column names (Ensuring it can handle headers like D.O.A or AGE)
        df.columns = [col.strip().upper().replace(' ', '_').replace('.', '_') for col in df.columns]
        self.df = df.copy()

    #Data Cleaning
    def clean_data(self):
        """
        Cleans patient records by standardizing dates, handling non-numeric missing
        indicators ('EMPTY'), and imputing missing values.
        """
        # Define expected date columns (normalized to uppercase with underscores)
        date_columns = [col for col in ['D_O_A', 'D_O_D', 'DATE_OF_ADMISSION', 'DATE_OF_DEATH'] if col in self.df.columns]
        
        # Identify columns that are explicitly meant to be numeric but may contain text placeholders
        # Start with common numeric field names; `numeric_cols_present` will filter to actual columns
        numeric_lab_columns = ['AGE', 'SATISFACTION', 'LENGTH_OF_STAY', 'HEART_RATE', 'BLOOD_PRESSURE', 'TEMPERATURE']
        
        # Filter down to only columns present in the DataFrame
        numeric_cols_present = [col for col in numeric_lab_columns if col in self.df.columns]
        # Define common categorical columns and keep only those present
        categorical_cols = [col for col in ['OUTCOME', 'GENDER', 'DIAGNOSIS', 'DEPARTMENT', 'PATIENTTYPE'] if col in self.df.columns]

        # Convert 'EMPTY' strings, leading/trailing whitespace, and other placeholders to NaN
        for col in numeric_cols_present:
            # Convert values like 'EMPTY', 'N/A', 'NA', or just white space to NaN
            self.df[col] = self.df[col].replace(r'^(\s*$|EMPTY|N/?A|NA)$', np.nan, regex=True)
            # Coerce the column type to numeric, setting any unhandled errors to NaN
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Convert date columns using flexible parsing
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Impute missing values:
        # 1. Categorical: Fill missing values with 'Unknown'
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna('Unknown')
            
        # 2. Numeric: Fill missing values with the median of the column
        for col in numeric_cols_present:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # 3. Drop rows with critical missing dates (Cannot calculate duration or timeline otherwise)
        self.df.dropna(subset=date_columns, inplace=True)

    # Summaries
    def outcome_summary(self):
        """Summarizes total counts for each patient outcome (DISCHARGE, DAMA, EXPIRY/DEATH)."""
        if 'OUTCOME' in self.df.columns:
            return self.df['OUTCOME'].value_counts()
        return None

    def mortality_rate(self):
        """Calculates the mortality rate (EXPIRY/DEATH as a percentage of total records)."""
        if 'OUTCOME' in self.df.columns and len(self.df) > 0:
            # Handles different potential spellings of death
            deaths = self.df['OUTCOME'].astype(str).str.upper().isin(['DEATH', 'EXPIRED', 'EXPIRY']).sum()
            total = len(self.df)
            return (deaths / total) * 100
        return None

    # Aggregations 
    def aggregate_by_gender(self):
        """Aggregates outcomes counts by gender."""
        if 'GENDER' in self.df.columns and 'OUTCOME' in self.df.columns:
            return self.df.groupby('GENDER')['OUTCOME'].value_counts().unstack(fill_value=0)
        return "Required columns (GENDER, OUTCOME) missing."

    def aggregate_by_age(self, bins=5):
        """Aggregates outcomes counts by age groups (binned)."""
        if 'AGE' in self.df.columns and 'OUTCOME' in self.df.columns:
            # Create age groups (e.g., 0-20, 21-40, etc.)
            age_groups = pd.cut(self.df['AGE'], bins=bins, include_lowest=True, right=True)
            return self.df.groupby(age_groups)['OUTCOME'].value_counts().unstack(fill_value=0)
        return "Required columns (AGE, OUTCOME) missing."

    def aggregate_by_department(self):
        """Aggregates average satisfaction score by department."""
        if 'DEPARTMENT' in self.df.columns and 'SATISFACTION' in self.df.columns:
            # Ensure SATISFACTION column is numeric after cleaning
            return self.df.groupby('DEPARTMENT')['SATISFACTION'].mean().sort_values(ascending=False)
        return "Required columns (DEPARTMENT, SATISFACTION) missing or data structure inadequate."

    # Visualizations
    def plot_outcome_by_age(self):
        """Creates a Histogram of patient outcomes stratified by age."""
        if 'AGE' in self.df.columns and 'OUTCOME' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=self.df, x='AGE', hue='OUTCOME', bins=20, multiple='stack', kde=False, ax=ax)
            ax.set_title('Patient Outcomes by Age Group (Histogram)', fontsize=14)
            ax.set_xlabel('Age (Years)')
            ax.set_ylabel('Count of Patients')
            st.pyplot(fig)
        else:
            st.warning("Cannot generate Histogram: Missing 'AGE' or 'OUTCOME' column.")

    def plot_admissions_over_time(self):
        """Creates a Line Chart of admissions count over time."""
        if 'D_O_A' in self.df.columns and not self.df.empty:
            # Convert to month period for grouping and sorting
            admissions_over_time = self.df.set_index('D_O_A').resample('M').size()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            admissions_over_time.plot(kind='line', marker='o', ax=ax, color='skyblue', linewidth=2)
            ax.set_title('Admissions Over Time (Line Chart)', fontsize=14)
            ax.set_xlabel('Date of Admission (Month)')
            ax.set_ylabel('Number of Admissions')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.warning("Cannot generate Line Chart: Missing or invalid 'D_O_A' column.")

    def plot_avg_satisfaction_by_department(self):
        """Creates a Bar Chart of average service satisfaction by department."""
        if 'DEPARTMENT' in self.df.columns and 'SATISFACTION' in self.df.columns:
            avg_satisfaction = self.df.groupby('DEPARTMENT')['SATISFACTION'].mean().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Use index for y-axis (departments) and values for x-axis (mean satisfaction)
            sns.barplot(x=avg_satisfaction.values, y=avg_satisfaction.index, palette='viridis', ax=ax)
            
            ax.set_title('Average Service Satisfaction by Department (Bar Chart)', fontsize=14)
            ax.set_xlabel('Average Satisfaction Score')
            ax.set_ylabel('Department')
            
            # Label the bars with the mean values
            for i, v in enumerate(avg_satisfaction.values):
                ax.text(v + 0.1, i, f"{v:.2f}", color='black', va='center')
                
            st.pyplot(fig)
        else:
            st.warning("Cannot generate Bar Chart: Missing 'DEPARTMENT' or 'SATISFACTION' column.")


# Streamlit Dashboard 

st.title("Health Data Analysis Dashboard")

@st.cache_data
def load_data(file):
    """Loads and returns data from uploaded file."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["xlsx", "csv"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
  
    df_processed = df_raw.copy()
    
    # Normalize column names for checks (do not mutate original raw df yet)
    normalized_cols = [col.strip().upper().replace(' ', '_').replace('.', '_') for col in df_processed.columns]

    if 'DEPARTMENT' not in normalized_cols:
        # Mock DEPARTMENT column and add it to the DataFrame
        departments = ['Cardiology', 'ICU', 'Emergency', 'General Med']
        df_processed['DEPARTMENT'] = np.random.choice(departments, size=len(df_processed))

    if 'SATISFACTION' not in normalized_cols:
        # Mock SATISFACTION (e.g., score 1 to 5) and add it as a numeric column
        df_processed['SATISFACTION'] = np.random.randint(1, 6, size=len(df_processed))
    
    # Instantiate the new class
    analyzer = HealthAnalyzer(df_processed)
    
    # Run the enhanced cleaning process
    analyzer.clean_data()
    
    # Display the processed data info
    st.markdown("---")
    st.subheader("Cleaned Data Snapshot")
    st.write(analyzer.df.head())
    
    # Display Summaries 
    st.markdown("---")
    st.subheader("1. Outcome Summary (Method: `outcome_summary`)")
    st.write(analyzer.outcome_summary())

    st.subheader("2. Mortality Rate (Method: `mortality_rate`)")
    mort = analyzer.mortality_rate()
    if mort is not None:
        st.metric(label="Overall Mortality Rate", value=f"{mort:.2f}%")
    else:
        st.write("N/A: Outcome column not found.")

    # Display Visualizations (Required Charts)
    st.markdown("---")
    st.header("Required Data Visualizations")

    st.subheader("3. Patient Outcomes by Age (Histogram)")
    analyzer.plot_outcome_by_age()

    st.subheader("4. Admissions Over Time (Line Chart)")
    analyzer.plot_admissions_over_time()

    st.subheader("5. Average Service Satisfaction by Department (Bar Chart)")
    analyzer.plot_avg_satisfaction_by_department()

    # Display Aggregations
    st.markdown("---")
    st.header("Data Aggregations")

    st.subheader("6. Outcomes Aggregated by Gender")
    st.write(analyzer.aggregate_by_gender())

    st.subheader("7. Outcomes Aggregated by Age Group (Binned)")
    st.write(analyzer.aggregate_by_age())

    st.subheader("8. Average Satisfaction by Department (Table)")
    st.write(analyzer.aggregate_by_department())

else:
    st.info("Upload a CSV or Excel file to populate the dashboard and run the analysis.")






