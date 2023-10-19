import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# Function to check the valid format


def is_valid_format(value):
    return re.match(r'^\s*\$?\d+(\.\d{2})?\s*$', str(value)) is not None

# Function to clean brand names


def clean_brand_name(brand):
    brand = brand.lower().strip()
    brand = brand.replace('&', 'and')
    return brand

# Function to perform data cleaning


def clean_data(df):
    # Remove trailing and leading spaces from string columns
    str_columns = df.select_dtypes(include=[object]).columns
    df[str_columns] = df[str_columns].apply(
        lambda x: x.str.strip() if x.dtype == "object" else x)

    # Remove extra whitespaces within string columns
    df[str_columns] = df[str_columns].apply(lambda x: x.str.replace(
        r'\s+', ' ', regex=True) if x.dtype == "object" else x)

    # Fill missing values in 'A&P' column with 0
    df['A&P'].fillna(0, inplace=True)

    df['Year'] = df['Year'].astype(
        int)  # Convert 'Year' to integers
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
    df = pd.get_dummies(df, columns=[
                        'Company', 'Brand', 'Month', 'Product'])
    X = df.drop(columns=['Budgeted'])  # Exclude 'Actual' from training data
    y = df['Budgeted']  # Only use 'Budgeted' as the target variable

    # Create an imputer to fill missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.7, random_state=42)

    forest_reg_budgeted = RandomForestRegressor(
        n_estimators=350, max_depth=11, min_samples_leaf=5, random_state=42)
    forest_reg_budgeted.fit(train_X, train_y)
    return forest_reg_budgeted, df.drop(columns=['Budgeted']).columns, train_X, train_y, test_X, test_y

# Function to predict Budgeted values


def predict_budgeted(model_budgeted, selected_data, columns_used_for_training):
    selected_data = pd.get_dummies(selected_data, columns=[
                                   'Company', 'Brand', 'Month', 'Product'])
    for col in columns_used_for_training:
        if col not in selected_data.columns:
            selected_data[col] = 0
    selected_data = selected_data[columns_used_for_training]
    predictions_budgeted = model_budgeted.predict(selected_data)
    return predictions_budgeted

# Function to calculate accuracy and score


def calculate_accuracy_and_score(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


st.set_page_config(layout="wide")
# Streamlit UI code
st.title("Budget Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # add A&P column if not there and with default 0.0
    if 'A&P' not in data.columns:
        data['A&P'] = "$0.0"

    data = clean_data(data)

    selected_company = st.selectbox(
        "Select a Company", data['Company'].unique())

    filtered_data = data[data['Company'] == selected_company]

    filtered_data.fillna(0, inplace=True)

    selected_brands = st.multiselect(
        "Select Brands", filtered_data['Brand'].unique(),
        # default=filtered_data["Brand"][0] if len(filtered_data["Brand"]) > 0 else "None"
    )

    filtered_data = filtered_data[filtered_data['Brand'].isin(selected_brands)]

    selected_products = st.multiselect(
        "Select Products", filtered_data['Product'].unique(),
        # default=filtered_data["Product"][0] if len(filtered_data["Product"]) > 0 else "None"
    )

    filtered_data = filtered_data[filtered_data['Product'].isin(
        selected_products)]

    # reindex filtered_data
    filtered_data = filtered_data.reset_index(drop=True)

    selected_months = st.multiselect("Select Months (Up to 3)", data['Month'].unique(), default=[
                                     "January", "February", "March"], key="selected_months", help="Select up to 3 months.", max_selections=3)

    # Ensure only up to 3 months are selected
    # if len(selected_months) > 3:
    #     st.warning("Please select up to 3 months.")
    #     selected_months = selected_months[:3]

    col1, col2, col3 = st.columns(3)  # Create 3 columns
    selected_year = col1.text_input("Enter Year", value="2019")

    # price_input = col2.text_input(
    #     "Enter Price (US/ TT)", value=filtered_data["Price (US/ TT)"][0] if len(filtered_data["Price (US/ TT)"]) > 0 else 0.0)
    # selected_prices = float(price_input) if price_input else 0.0

    # selected_ap = col3.text_input("Enter A&P", value=filtered_data["A&P"][0] if len(
    #     filtered_data["A&P"]) > 0 else 0.0)
    # selected_ap = float(selected_ap) if selected_ap else 0.0

    model_budgeted, columns_used_for_training, train_X, train_y, test_X, test_y = train_model(
        data)

    selected_data = filtered_data.copy()
    selected_data['Year'] = selected_year
    # selected_data['Price (US/ TT)'] = selected_prices
    # selected_data['A&P'] = selected_ap

    # Remove duplicates based on the columns used for selection
    selected_data = selected_data.drop_duplicates(
        subset=['Company', 'Brand', 'Product', 'Month'])

    # Filter the selected data for the chosen months
    selected_data = selected_data[selected_data['Month'].isin(selected_months)]

    selected_data_copy = selected_data.copy()  # for futher use

    # predictions_budgeted = predict_budgeted(
    #     model_budgeted, selected_data, columns_used_for_training)

    # selected_data = selected_data.drop(columns=['Actual', 'Dummy'])

    # selected_data['Predicted Budget'] = predictions_budgeted
    # selected_data['Predicted Budget'] = selected_data['Predicted Budget'].apply(
    #     lambda x: round(x, 2))

    st.divider()
    st.markdown(" ##### Edit the value of `Price (US/ TT)` and `A&P` ")

    col1, col2 = st.columns([3, 1])

    # selected_data['Actual'] = selected_data['Actual'].apply(
    #     lambda x: round(x, 2))

    editable_data = col1.experimental_data_editor(selected_data[[
        'Company', 'Brand', 'Product', 'Month', 'Year', 'Price (US/ TT)', 'A&P', 'Actual']], use_container_width=True)

   # Create a new DataFrame with the same column order as the training data
    updated_data = pd.DataFrame()

    # Add 'Year', 'Price (US/ TT)', and 'A&P' columns to the updated_data
    updated_data['Year'] = selected_data['Year']
    updated_data['Price (US/ TT)'] = editable_data['Price (US/ TT)']
    updated_data['A&P'] = editable_data['A&P']

    # selected_data['Price (US/ TT)'] = selected_data['Price (US/ TT)']
    # selected_data['A&P'] = selected_data['A&P']
    selected_data_copy['Price (US/ TT)'] = updated_data['Price (US/ TT)']
    selected_data_copy['A&P'] = updated_data['A&P']

    # Recalculate the predicted budget values based on the updated values
    updated_data['Predicted Budget'] = predict_budgeted(
        model_budgeted, selected_data_copy, columns_used_for_training)

    # if st.button('Recalculate Predicted Budget'):
    # Display the updated DataFrame with the new predicted budget values
    # Show a DataFrame with 'Company', 'Brand', 'Product', 'Month', 'Year' from selected_data
    # and 'Price (US/ TT)', 'A&P', 'Predicted Budget' from updated_data

    selected_columns = [
        # 'Company', 'Brand', 'Product', 'Month', 'Year'
    ]
    updated_columns = [
        # 'Price (US/ TT)', 'A&P',
        'Predicted Budget'
    ]

    # display_data = pd.concat(
    #     [selected_data[selected_columns], updated_data[updated_columns]], axis=1)

    display_data = updated_data['Predicted Budget']

    # Predicted Budget as 2 decimal point

    display_data = display_data.apply(lambda x: round(x, 2))

    col2.dataframe(display_data)

    # make a download button for the result dataframe

    # merge editable_data and display_data

    final_data = pd.concat([editable_data, display_data], axis=1)

    st.download_button(
        label="Download Result DataFrame (Budgeted)",
        data=final_data.to_csv(index=False),
        file_name="result_dataframe_budgeted.csv",
        mime="text/csv",
    )
    # ------------- End of Result DataFrame (Budgeted) ------------- #

    # ------------- Start of Accuracy and Score ------------- #

    st.divider()
    st.subheader("Predicted Budget Accuracy and Score")
    # Use 'Predicted Budget' column
    y_true = selected_data_copy['Budgeted']
    # Use 'Predicted Budget' column
    y_pred = updated_data['Predicted Budget']

    mae, mse, r2 = calculate_accuracy_and_score(y_true, y_pred)
    # st.write(f"Mean Absolute Error: {mae:.2f}")
    # st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    forest_train_score = model_budgeted.score(train_X, train_y)
    forest_test_score = model_budgeted.score(test_X, test_y)

    st.write("Training Score:", forest_train_score)
    st.write("Testing Score:", forest_test_score)

    import matplotlib.pyplot as plt  # Add this line

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        # 'Company': selected_data_copy['Company'],
        # 'Brand': selected_data_copy['Brand'],
        # 'Product': selected_data_copy['Product'],
        'Month': selected_data_copy['Month'],
        'Actual': selected_data_copy['Actual'],
        'Predicted': final_data['Predicted Budget']
    })

    # Group by month
    results_grouped = results_df.groupby(
        'Month').max()

    # Create a bar chart to visualize the data
    fig, ax = plt.subplots()
    results_grouped.plot(kind='bar', ax=ax)
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.title(f"Actual vs Predicted Values for {selected_year}")
    st.pyplot(fig)
