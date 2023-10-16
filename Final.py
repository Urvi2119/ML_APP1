import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
import random
import re

# def is_valid_format(value):
#     return re.match(r'^\s*\$\s*\d+\.\d{2}\s*$', str(value)) is not None


def is_valid_format(value):
    return re.match(r'^\s*\$?\d+(\.\d{2})?\s*$', str(value)) is not None


def clean_brand_name(brand):
    # Example data cleaning steps:
    # 1. Convert to lowercase
    # 2. Remove leading and trailing spaces
    # 3. Handle special characters or variations
    brand = brand.lower().strip()

    # Additional cleaning steps if needed
    # Example: Replace '&' with 'and'
    brand = brand.replace('&', 'and')

    return brand
    # Remove spaces and special characters, replace with underscores
    # cleaned_name = re.sub(r'[^a-zA-Z0-9]', '_', brand_name)
    # return cleaned_name


# Global variables
forest_bud = 0
forest_pre = 0
st.title("Welcome")

try:
    # File upload from here
    uploaded_file = st.file_uploader(" ", type={"csv", "txt"})  # type: ignore
    if uploaded_file is not None:
        uploaded_file_df = pd.read_csv(uploaded_file)
        uploaded_file_df_orig = uploaded_file_df
    uploaded_file_df['Year'] = uploaded_file_df['Year'].astype(str)
    uploaded_file_df['Year'] = uploaded_file_df['Year'].str.replace(",", "")
    uploaded_file_df['A&P'] = uploaded_file_df['A&P'].astype(str)
    uploaded_file_df['A&P'] = uploaded_file_df['A&P'].str.replace("$", "")
    st.write(uploaded_file_df)

    columns_to_add = ['Price (US/ TT)', 'A&P']

    # Remove leading/trailing spaces from column names
    uploaded_file_df.columns = uploaded_file_df.columns.str.strip()

    uploaded_file_df_1 = uploaded_file_df  # Create a copy of the DataFrame

    for column in columns_to_add:
        if column not in uploaded_file_df_1.columns:
            uploaded_file_df_1[column] = 0  # Add the column with None values

    uploaded_file_df_1 = uploaded_file_df_1.drop(columns=['Dummy'])

    # Filling the missing values with mean
    uploaded_file_df_1["Actual"] = uploaded_file_df_1["Actual"].fillna(
        uploaded_file_df_1["Actual"].mean())
    uploaded_file_df_1.isna().sum()
    uploaded_file_df_1["A&P"] = uploaded_file_df_1["A&P"].fillna(0.0)
    # Clean the "A&P" column using regular expressions and replace problematic values with 0
    uploaded_file_df_1["A&P"] = uploaded_file_df_1["A&P"].apply(
        lambda x: '0.0' if not is_valid_format(x) else x)

    uploaded_file_df_1["A&P"] = uploaded_file_df_1['A&P'].astype(str)

    # Clean the 'A&P' column
    uploaded_file_df_1["A&P"] = uploaded_file_df_1["A&P"].apply(
        lambda x: '0.0' if not re.match(r'^\d+(\.\d+)?$', str(x)) else x)

    # Convert 'A&P' column to float type
    uploaded_file_df_1["A&P"] = uploaded_file_df_1["A&P"].str.replace(
        '[^\d.]', '', regex=True).astype(float)
    # Remove '$' from "Price (US/ TT)" column
    uploaded_file_df_1["Price (US/ TT)"] = uploaded_file_df_1["Price (US/ TT)"].str.replace(
        '[\$,]+', '', regex=True)
    uploaded_file_df_1["Price (US/ TT)"] = uploaded_file_df_1["Price (US/ TT)"].astype(float)
    uploaded_file_df_1 = pd.get_dummies(uploaded_file_df_1, columns=[
        'Company', 'Brand', 'Month', 'Product', 'Price (US/ TT)', 'A&P'])
    st.subheader("Clean Data:")
    st.write(uploaded_file_df_1)

    # Splitting the data into training and testing modules
    X = uploaded_file_df_1.drop(columns=['Actual', 'Budgeted'])
    y = uploaded_file_df_1['Actual']
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.3, random_state=42)

    st.subheader("Training Data :")
    st.write(train_X)

    st.subheader("Test Data :")
    st.write(test_X)

    # Implementing Random Forest with suggested changes
    forest_reg = RandomForestRegressor(
        n_estimators=350, max_depth=11, min_samples_leaf=5, random_state=42)  # Adjust hyperparameters as needed

    # Fit the model
    forest_reg.fit(train_X, train_y)

    # Fit the model
    # forest_reg.fit(X, y)  # Train the model with all possible feature values

    # Testing and training accuracy
    forest_train_score = forest_reg.score(train_X, train_y)
    forest_test_score = forest_reg.score(test_X, test_y)

    st.subheader("Train Accuracy :")
    st.write("Random Forest:", forest_train_score)

    st.subheader("Test Accuracy :")
    st.write("Random Forest:", forest_test_score)

    lst = []
    result_temp = []
    # Predict the values for the uploaded file
    for index, row in uploaded_file_df.iterrows():
        lst_temp = []
        # Prepare the input data for the model
        Company_name = str(row['Company'])
        Brand_name = str(row['Brand'])
        Product_value = str(row['Product'])
        Month_name = str(row['Month'])
        Year_value = str(row['Year'])
        Actual = str(row['Actual'])
        Price = str(row['Price (US/ TT)'])
        AP = str(row['A&P'])

        Company = 'Company_' + Company_name
        Brand = 'Brand_' + Brand_name
        Product = 'Product_' + Product_value
        Month = 'Month_' + Month_name

        # Create a new dataframe with the input data
        new_data = pd.DataFrame({f'{Company}': 1,
                                 f'{Brand}': 1,
                                 f'{Product}': 1,
                                 f'{Month}': 1,
                                 'Year': int(Year_value),
                                 f'Price (US/ TT)': 1,
                                 f'A&P': 1,
                                 }, index=[0])

        # Preprocess the new data
        new_data = pd.get_dummies(new_data, drop_first=True)

        # Reindex the columns to match the training data
        new_data = new_data.reindex(columns=train_X.columns, fill_value=0)

        # Predict the value using Random Forest
        forest_prediction = forest_reg.predict(new_data)[0]

        # Add the predicted value to the list
        lst_temp.append(str(forest_prediction))
        lst_temp.append(str(row['Actual']))
        lst_temp.append(str(row['Budgeted']))
        lst_temp.append(str(row['Price (US/ TT)']))
        lst_temp.append(str(row['A&P']))
        lst.append(lst_temp)

        result_temp.append({
            'Company': Company.replace('Company_', ' '),
            'Brand': Brand.replace('Brand_', ' '),
            'Product': Product.replace('Product_', ' '),
            'Month': Month.replace('Month_', ' '),
            'Year': str(Year_value).replace(',', ''),
            'Prediction': forest_prediction,
            'Actual': Actual,
            'Price (US/ TT)':  Price,
            'A&P': AP,

        })

    company_list = uploaded_file_df['Company'].unique()
    product_list = uploaded_file_df['Product'].unique()
    month_list = uploaded_file_df['Month'].unique()
    month_list = np.delete(month_list, 11)
    actual_list = uploaded_file_df['Actual'].unique()
    brand_list = uploaded_file_df['Brand'].unique()
    unique_prices = uploaded_file_df['Price (US/ TT)'].unique()
    unique_ap_values = uploaded_file_df['A&P'].unique()

    company_and_brand_Dict = {}
    brand_product_dict = {}
    for company in company_list:
        related_brands = uploaded_file_df.loc[uploaded_file_df['Company'] == company, 'Brand'].unique(
        )
        company_and_brand_Dict[company] = related_brands
        brand_product_dict[company] = {}
        for brand in related_brands:
            related_products = uploaded_file_df.loc[(uploaded_file_df['Company'] == company) & (
                uploaded_file_df['Brand'] == brand), 'Product'].unique()
            brand_product_dict[company][brand] = related_products

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(result_temp)
    st.subheader("Prediction for existing periods :")
    st.write(results_df)

    st.subheader("Graph for random value comparison :")
    # Collecting data for Graph for comparison
    random_value = st.selectbox(
        "Random values : ", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    random_sample = random.sample(lst, k=random_value)

    # Creating a graph of random values
    lst_pre_forest = []
    lst_act = []
    lst_bud = []
    for i in range(0, random_value):
        lst_pre_forest.append(random_sample[i][0])
        lst_act.append(random_sample[i][1])
        lst_bud.append(random_sample[i][2])

    x_rand_lforest = lst_pre_forest
    y_rand = lst_act
    traildf = {"Actual": y_rand,
               "Budgeted": lst_bud, "Predicted Random Forest": x_rand_lforest}
    st.write(f"Random {random_value} values from the dataset")
    st.line_chart(data=traildf)

    # Create multi-select fields for selecting multiple Companies, Brands, and Products
    st.subheader("Prediction for new periods :")
    st.subheader('Please select Companies, Brands, Products, and Year:')

    selected_company = st.selectbox('Company', company_list)
    to_show_brand = []

    for brand in company_and_brand_Dict[selected_company]:
        to_show_brand.append(brand)

    selected_brand = st.multiselect("Brands", to_show_brand)
    selected_brand = set(selected_brand)
    len_list = []

    brand_product_dicti = {}
    to_show_products = []

    related_brands = company_and_brand_Dict[selected_company]
    for brand in related_brands:
        if brand in tuple(selected_brand):
            for prod in brand_product_dict[selected_company][brand]:
                if prod not in to_show_products:
                    to_show_products.append(prod)

    selected_product = st.multiselect("Products", to_show_products)
    selected_month = st.multiselect("Months", month_list)
    if len(selected_month) > 3:
        st.warning("Please select only three months.")
        selected_month = selected_month[:3]
    input_year = st.text_input('Year')

    # selected_price = [st.selectbox(
    #     'Price (US/ TT)', unique_prices,  label_visibility="hidden")]
    # selected_ap_values = [st.selectbox(
    #     'A&P', unique_ap_values,  label_visibility="hidden", index=1)]

    # Collect the columns used for training the model
    training_columns = train_X.columns

    results = []
    selectedcompany = selected_company
    selectedbrand = selected_brand
    selectedproduct = selected_product
    # Generate predictions for all selected combinations
    for month in selected_month:

        for selected_brand in selectedbrand:
            for selected_product in selectedproduct:
                input_Company = 'Company_' + selected_company
                input_Brand = 'Brand_' + selected_brand
                input_Product = 'Product_' + selected_product
                input_Month = 'Month_' .join(month)

                new_data = pd.DataFrame({f'{input_Company}': 1,
                                        f'{input_Brand}': 1,
                                         f'{input_Product}': 1,
                                         # f'{input_Month}': 1,
                                         'Year': int(input_year)}, index=[0])

                # Preprocess the new data
                # new_data = pd.get_dummies(new_data, drop_first=True)

                # Reindex the columns to match the training data
                # new_data = new_data.reindex(
                #     columns=train_X.columns, fill_value=0)

                # # Add selected Price(US/TT) columns
                # for price_option in selected_price:
                #     if price_option != 'Empty':
                #         price_option = str(price_option)
                #         price_option = price_option.split('$')
                #         price_option = str(price_option[1])
                #         new_data[f'Price (US/ TT)_{price_option}'] = 1

                # # Add selected A&P columns
                # for ap in selected_ap_values:
                #     if ap != 'Empty':
                #         ap = str(ap)
                #         new_data[f'A&P_{ap}'] = 1

                # Predict the value using Random Forest
                # forest_prediction = forest_reg.predict(new_data)[0]

                # Append the results to the list
                results.append({
                    'Company': input_Company.replace('Company_', ' '),
                    'Brand': input_Brand.replace('Brand_', ' '),
                    'Product': input_Product.replace('Product_', ' '),
                    'Month': month.replace('Month_', ' '),

                    "Year": str(input_year),

                    # 'Price (US/ TT)': ', '.join(map(str, selected_price)),
                    # 'A&P': ', '.join(map(str, selected_ap_values)),
                    'Price (US/ TT)': '$0.00',
                    'A&P': 0.00,
                    # 'Prediction': 0
                })

    # Convert the results list to a dataframe
    df_results = pd.DataFrame(results)

    # Save the dataframe as a CSV file
    df_results.to_csv('results.csv', index=False)

    st.subheader("Prediction for new period :")

    # Make the DataFrame editable
    edited_df = st.experimental_data_editor(df_results)

    st.download_button(
        label="Download data as CSV",
        data=edited_df.to_csv().encode('utf-8'),
        file_name='large_df.csv',
        mime='text/csv',
    )

    # Read the uploaded CSV file into a DataFrame
    # uploaded_csv = st.file_uploader("Upload your CSV file here", type="csv")
    if st.button('Recalculate Prediction'):
        # Extract the 'Price (US/ TT)' and 'A&P' columns from the edited_df
        edited_prices = edited_df['Price (US/ TT)']
        edited_ap = edited_df['A&P']

        # Update the 'Price (US/ TT)' and 'A&P' columns in uploaded_file_df_1 with the edited values
        uploaded_file_df_1['Price (US/ TT)'] = edited_prices
        uploaded_file_df_1['A&P'] = edited_ap

        results2 = []

        for index, (row_edited, row_uploaded) in enumerate(zip(edited_df.iterrows(), uploaded_file_df_1.iterrows())):
            # Extract the 'Company' value from the edited DataFrame
            # Use the 'Company' value from the edited DataFrame
            Company_name = row_edited[1]['Company']

            Brand_name = 'Brand_' + row_edited[1]['Brand']
            Product_value = 'Product_' + row_edited[1]['Product']
            Month_name = 'Month_' + row_edited[1]['Month']
            Year_value = row_edited[1]['Year']
            Actual = row_uploaded[1]['Actual']
            Price = row_edited[1]['Price (US/ TT)']
            A_P = row_edited[1]['A&P']

            # Create a new DataFrame with the input data
            new_data = pd.DataFrame({
                f'Company_{Company_name}': 1,
                f'{Brand_name}': 1,
                f'{Product_value}': 1,
                f'{Month_name}': 1,
                'Year': int(Year_value),
                'Price (US/ TT)': row_uploaded[1]['Price (US/ TT)'],
                'A&P': row_uploaded[1]['A&P'],
            }, index=[0])

            # Preprocess the new data
            new_data = pd.get_dummies(new_data, drop_first=True)

            # Reindex the columns to match the training data
            new_data = new_data.reindex(columns=train_X.columns, fill_value=0)

            # Predict the value using Random Forest
            forest_prediction = forest_reg.predict(new_data)[0]

            # Append the results to the list
            results2.append({
                'Company': Company_name,  # Use the 'Company' value from the edited DataFrame
                'Brand': Brand_name,
                'Product': Product_value,
                'Month': Month_name,
                'Year': Year_value,
                'Price (US/ TT)': Price,
                'A&P': A_P,
                'Prediction': forest_prediction
            })

        # Convert the results list to a DataFrame
        df_results2 = pd.DataFrame(results2)

        # Display the recalculated predictions
        st.subheader("Recalculated Prediction:")
        st.write(df_results2)
        st.header("Thank you!")

# except Exception as e:
#     if str(e) != "name 'uploaded_file_df' is not defined":
#         st.error(f"An error occurred: {str(e)}")

except:
    pass

# killall streamlit
# nohup streamlit run app_final.py
