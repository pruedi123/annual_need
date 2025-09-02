import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Step 1: Simulate the ending values for $1 invested each year for N years
def simulate_ending_value_index(df_column, num_years, row_increment):
    results = []
    for start_row in range(0, len(df_column) - (row_increment * (num_years - 1))):
        investment_value = 0  # Start with $0 since we are adding $1 each year
        for year in range(num_years):
            current_row = start_row + (row_increment * year)
            factor = df_column[current_row]
            investment_value = (investment_value + 1) * factor  # $1 added each year
        results.append(investment_value)
    return results

# Step 2: Calculate the required investment based on the ending value index
def calculate_required_investment(ending_values, goal, confidence_level):
    sorted_values = sorted(ending_values)
    index = int((1 - confidence_level) * len(sorted_values))  # Find the index for the confidence level
    ending_value_at_confidence = sorted_values[index]
    required_investment_per_dollar = 1 / ending_value_at_confidence
    return required_investment_per_dollar

# Initialize df to None to ensure it's always defined
df = None

# Load the Excel file with historical return factors
file_path = 'all_portfolio_annual_factor_20_bps.xlsx'
sheet_name = 'allocation_factors'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    st.write("Data loaded successfully!")  # Debugging message to ensure data is loaded
except Exception as e:
    st.error(f"Error loading data: {e}")  # Display error if loading fails

# Streamlit interface for selecting parameters
st.title("Annual Investment Calculator")
st.write("Select parameters to calculate the required annual investment:")

# Streamlit sliders for interactive input
goal = st.slider("Goal ($)", min_value=10000, max_value=1000000, value=1000000, step=10000)
num_years = st.slider("Number of Years", min_value=1, max_value=41, value=30)
confidence_level = st.slider("Confidence Level (%)", min_value=50, max_value=100, value=90) / 100
row_increment = 12  # Assuming monthly data, skip 12 rows for annual factors

# Display initial message to ensure the page is not blank
st.write("Press the 'Calculate' button to see the required annual investment.")

# Button to trigger calculation
if st.button("Calculate"):
    if df is None:
        st.write("No data available to process. Please check the data file.")  # Error if df is not loaded
    else:
        # Loop through each allocation and calculate the required annual investment
        results_summary = []
        for column_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[column_name]):
                
                # Step 1: Simulate the ending value index for this allocation
                df_column = df[column_name]
                ending_values = simulate_ending_value_index(df_column, num_years, row_increment)
                
                # Step 2: Calculate required investment based on the ending value index
                required_investment_per_dollar = calculate_required_investment(ending_values, goal, confidence_level)
                required_annual_investment = required_investment_per_dollar * goal
                
                # Store the results for this allocation, converting to integers (no decimals)
                results_summary.append({
                    'Allocation': column_name,
                    'Required Annual Investment': int(required_annual_investment)  # No decimals
                })
        
        # Convert results to a DataFrame for easy viewing
        results_df = pd.DataFrame(results_summary)

        # Define the specific order for the allocations
        allocation_order = [
            'LBM 100E', 'LBM 90E', 'LBM 80E', 'LBM 70E', 'LBM 60E', 'LBM 50E',
            'LBM 40E', 'LBM 30E', 'LBM 20E', 'LBM 10E', 'LBM 100F'
        ]

        # Create a mapping for renaming the allocations
        allocation_mapping = {
            'LBM 100E': '100% Equity',
            'LBM 90E': '90% Equity',
            'LBM 80E': '80% Equity',
            'LBM 70E': '70% Equity',
            'LBM 60E': '60% Equity',
            'LBM 50E': '50% Equity',
            'LBM 40E': '40% Equity',
            'LBM 30E': '30% Equity',
            'LBM 20E': '20% Equity',
            'LBM 10E': '10% Equity',
            'LBM 100F': '100% Fixed'
        }

        # Ensure the DataFrame is ordered according to this custom list and apply the mapping
        results_df['Allocation'] = pd.Categorical(results_df['Allocation'], categories=allocation_order, ordered=True)
        results_df = results_df.sort_values('Allocation')
        results_df['Allocation'] = results_df['Allocation'].map(allocation_mapping)

        # Display the DataFrame in Streamlit
        st.write("Results--Real, inflation adjusted values (in 'todays' dollars):")
        st.write(results_df)

        # Find the index of the minimum value
        min_value = results_df['Required Annual Investment'].min()
        colors = ['green' if v == min_value else 'blue' for v in results_df['Required Annual Investment']]

        # Create an interactive chart using Plotly with custom order respected and conditional coloring
        st.write("Interactive Chart:")
        fig = go.Figure(data=[go.Bar(
            x=results_df['Allocation'],
            y=results_df['Required Annual Investment'],
            marker_color=colors  # Use the conditional coloring
        )])

        fig.update_layout(title="Required Annual Investment by Allocation", xaxis_title="Allocation", yaxis_title="Required Annual Investment")
        st.plotly_chart(fig)

        # Option to download the DataFrame as an Excel file
        output_file = 'required_annual_investment_by_allocation.xlsx'
        results_df.to_excel(output_file, index=False)
        #st.write(f"Required annual investments for all allocations have been saved to {output_file}.")

        # Download button for CSV version
        st.download_button(
            label="Download Results as CSV",
            data=results_df.to_csv(index=False),
            file_name='required_annual_investment_by_allocation.csv',
            mime='text/csv'
        )

import streamlit as st

st.markdown('[Click here to go to Main Site](https://www.paulruedi.com)')