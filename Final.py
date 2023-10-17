import pandas as pd
import re
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function to check the valid format

if 'historical_ap_values' not in st.session_state:
    st.session_state.historical_ap_values = []


def is_valid_format(value):
    return re.match(r'^\s*\$?\d+(\.\d{2})?\s*$', str(value)) is not None

# Function to clean brand names


def clean_brand_name(brand):
    brand = brand.lower().strip()
    brand = brand.replace('&', 'and')
    return brand

# Function to perform data cleaning


def clean_data(df):
    df['Year'] = df['Year'].astype(str).str.replace(",", "")
    df['A&P'] = df['A&P'].str.replace("$", "")
    df['Price (US/ TT)'] = df['Price (US/ TT)'].str.replace('[\$,]+',
                                                            '', regex=True)
    df['A&P'] = df['A&P'].apply(
        lambda x: '0.0' if not is_valid_format(x) else x)
    df['A&P'] = df['A&P'].str.replace('[^\d.]', '', regex=True).astype(float)
    df['Price (US/ TT)'] = df['Price (US/ TT)'].astype(float)
    return df

# Function to train the Random Forest model


def train_model(df):
    df = pd.get_dummies(df, columns=['Company', 'Brand', 'Month', 'Product'])
    X = df.drop(columns=['Actual', 'Budgeted'])
    y = df[['Actual', 'Budgeted']]
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.3, random_state=42)
    train_y['Actual'] = train_y['Actual'].fillna(train_y['Actual'].mean())
    train_y['Budgeted'] = train_y['Budgeted'].fillna(
        train_y['Budgeted'].mean())
    forest_reg_actual = RandomForestRegressor(
        n_estimators=350, max_depth=11, min_samples_leaf=5, random_state=42)
    forest_reg_budgeted = RandomForestRegressor(
        n_estimators=350, max_depth=11, min_samples_leaf=5, random_state=42)
    forest_reg_actual.fit(train_X, train_y['Actual'])
    forest_reg_budgeted.fit(train_X, train_y['Budgeted'])
    return forest_reg_actual, forest_reg_budgeted, train_X.columns.tolist()

# Function to predict Actual and Budgeted values


def predict_values(model_actual, model_budgeted, selected_data, columns_used_for_training):
    selected_data = pd.get_dummies(selected_data, columns=[
                                   'Company', 'Brand', 'Month', 'Product'])
    for col in columns_used_for_training:
        if col not in selected_data.columns:
            selected_data[col] = 0
    selected_data = selected_data[columns_used_for_training]
    predictions_actual = model_actual.predict(selected_data)
    predictions_budgeted = model_budgeted.predict(selected_data)
    return predictions_actual, predictions_budgeted


# Streamlit UI code
st.title("Budget Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = clean_data(data)

    # Initialize the list to store historical A&P values

    selected_company = st.selectbox(
        "Select a Company", data['Company'].unique())

    filtered_data = data[data['Company'] == selected_company]

    selected_brands = st.multiselect(
        "Select Brands", filtered_data['Brand'].unique())

    filtered_data = filtered_data[filtered_data['Brand'].isin(selected_brands)]

    selected_products = st.multiselect(
        "Select Products", filtered_data['Product'].unique())

    filtered_data = filtered_data[filtered_data['Product'].isin(
        selected_products)]

    # reindex filtered_data
    filtered_data = filtered_data.reset_index(drop=True)

    selected_months = st.multiselect(
        "Select Months (Up to 3)", data['Month'].unique(), default=["January", "February", "March"], key="selected_months", help="Select up to 3 months."
    )

    # Ensure only up to 3 months are selected
    if len(selected_months) > 3:
        st.warning("Please select up to 3 months.")
        selected_months = selected_months[:3]

    col1, col2, col3 = st.columns(3)  # Create 3 columns
    selected_year = col1.text_input("Enter Year", value="2024")

    # st.write(filtered_data["Price (US/ TT)"])

    price_input = col2.text_input(
        "Enter Price (US/ TT)", value=filtered_data["Price (US/ TT)"][0] if len(filtered_data["Price (US/ TT)"]) > 0 else 0.0)
    selected_prices = float(price_input) if price_input else 0.0

    selected_ap = col3.text_input(
        "Enter A&P", value=filtered_data["A&P"][0] if len(filtered_data["A&P"]) > 0 else 0.0)
    selected_ap = float(selected_ap) if selected_ap else 0.0

    if selected_ap is not None:
        his = st.session_state.historical_ap_values
        his.append(selected_ap)
        st.session_state.historical_ap_values = his
        # st.write(his)

        col1, col2 = st.columns(2)  # Create two columns

        if col1.button("Show Historical A&P Chart"):
            if his:
                st.subheader("Historical A&P Values Chart")
                st.line_chart(his)

        if col2.button("Reset Chart"):
            st.session_state.historical_ap_values = []
            # st.write(st.session_state.historical_ap_values)
            # st.subheader("Historical A&P Values Chart")
            # st.line_chart(st.session_state.historical_ap_values)

    # # Append the selected A&P value to the historical values
    # historical_ap_values.append(selected_ap)

    model_actual, model_budgeted, columns_used_for_training = train_model(data)

    selected_data = filtered_data.copy()
    selected_data['Year'] = selected_year
    selected_data['Price (US/ TT)'] = selected_prices
    selected_data['A&P'] = selected_ap

    # Remove duplicates based on the columns used for selection
    selected_data = selected_data.drop_duplicates(
        subset=['Company', 'Brand', 'Product', 'Month'])

    # Filter the selected data for the chosen months
    selected_data = selected_data[selected_data['Month'].isin(selected_months)]

    predictions_actual, predictions_budgeted = predict_values(
        model_actual, model_budgeted, selected_data, columns_used_for_training)

    selected_data = selected_data.drop(columns=['Actual', 'Budgeted', 'Dummy'])

    selected_data['Predicted Actual'] = predictions_actual
    selected_data['Predicted Actual'] = selected_data['Predicted Actual'].apply(
        lambda x: round(x, 2))

    selected_data['Predicted Budgeted'] = predictions_budgeted
    selected_data['Predicted Budgeted'] = selected_data['Predicted Budgeted'].apply(
        lambda x: round(x, 2))

    st.subheader("Result DataFrame (Actual):")
    st.dataframe(selected_data[['Company', 'Brand', 'Product', 'Month',
                 'Year', 'Price (US/ TT)', 'A&P', 'Predicted Actual']])
    st.subheader("Result DataFrame (Budgeted):")
    st.dataframe(selected_data[['Company', 'Brand', 'Product', 'Month',
                 'Year', 'Price (US/ TT)', 'A&P', 'Predicted Budgeted']])

    # # Create a chart to display historical A&P values
    # st.subheader("Historical A&P Values Chart")
    # st.line_chart(historical_ap_values)
