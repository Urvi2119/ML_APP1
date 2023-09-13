import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regressor
import random

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
    st.write(uploaded_file_df)
    columns = ['Company', 'Brand', 'Product',
               'Actual', 'Month', 'Year', 'Budgeted']
    uploaded_file_df_1 = uploaded_file_df[columns]

    # Converting Categorical data into Numerical data
    uploaded_file_df_1 = pd.get_dummies(uploaded_file_df_1, columns=[
        'Company', 'Brand', 'Month', 'Product'])
    st.subheader("Clean Data:")

    # Filling the missing values with mean
    uploaded_file_df_1["Actual"] = uploaded_file_df_1["Actual"].fillna(
        uploaded_file_df_1["Actual"].mean())
    uploaded_file_df_1.isna().sum()
    st.write(uploaded_file_df_1)

    # Converting the Year column into integer
    uploaded_file_df['Year'] = uploaded_file_df['Year'].astype(int)

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
        Company_name = row['Company']
        Brand_name = row['Brand']
        Product_value = row['Product']
        Month_name = row['Month']
        Year_value = row['Year']
        Actual = row['Actual']

        Company = 'Company_' + Company_name
        Brand = 'Brand_' + Brand_name
        Product = 'Product_' + Product_value
        Month = 'Month_' + Month_name

        # Create a new dataframe with the input data
        new_data = pd.DataFrame({f'{Company}': [1],
                                 f'{Brand}': [1],
                                 f'{Product}': [1],
                                 f'{Month}': [1],
                                 'Year': [int(Year_value)]})

        # Preprocess the new data
        new_data = pd.get_dummies(new_data, drop_first=True)

        # Reindex the columns to match the training data
        new_data = new_data.reindex(columns=train_X.columns, fill_value=0)

        # Predict the value using Random Forest
        forest_prediction = forest_reg.predict(new_data)[0]

        # Add the predicted value to the list
        lst_temp.append(forest_prediction)
        lst_temp.append(row['Actual'])
        lst_temp.append(row['Budgeted'])
        lst.append(lst_temp)

        result_temp.append({
            'Company': Company.replace('Company_', ' '),
            'Brand': Brand.replace('Brand_', ' '),
            'Product': Product.replace('Product_', ' '),
            'Month': Month.replace('Month_', ' '),
            'Year': str(Year_value).replace(',', ''),
            'Prediction': forest_prediction,
            'Actual': Actual,
        })

    company_list = uploaded_file_df['Company'].unique()
    product_list = uploaded_file_df['Product'].unique()
    month_list = uploaded_file_df['Month'].unique()
    month_list = np.delete(month_list, 12)
    actual_list = uploaded_file_df['Actual'].unique()
    brand_list = uploaded_file_df['Brand'].unique()
    company_and_brand_Dict = {}
    brand_product_dict = {}
    for company in company_list:
        related_brands = uploaded_file_df.loc[uploaded_file_df['Company']
                                            == company, 'Brand'].unique()
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

    # Created selection fields for the user to select the Company, Brand, Product, Month and Year
    st.subheader("Prediction for new periods :")
    st.subheader(
        'Please select a Company and a Year for Brands, Products, and Months :')
    selected_company = st.selectbox('Company', company_list)
    to_show_brand = company_and_brand_Dict[selected_company]
    selected_brand = st.multiselect("Brands", to_show_brand)
    len_list = []
    for i in range(len(selected_brand)):
        len_list.append(
            list(brand_product_dict[selected_company][selected_brand[i]]))

    max_index = max(range(len(len_list)), key=lambda i: len(len_list[i]))
    to_show_product = brand_product_dict[selected_company][selected_brand[max_index]]
    selected_product = st.multiselect("Products", to_show_product)
    selected_month = st.multiselect("Months", month_list)
    input_year = st.text_input('Year')
    results = []

    for brand in selected_brand:
        for product in selected_product:
            for month in selected_month:
                input_Company = 'Company_' + selected_company
                input_Brand = 'Brand_' + brand
                input_Product = 'Product_' + product
                input_Month = 'Month_' + month

                new_data = pd.DataFrame({f'{input_Company}': [1],
                                         f'{input_Brand}': [1],
                                         f'{input_Product}': [1],
                                         f'{input_Month}': [1],
                                         'Year': [int(input_year)]})
                new_data = pd.get_dummies(new_data, drop_first=True)
                # Reindex the columns to match the training data
                new_data = new_data.reindex(
                    columns=train_X.columns, fill_value=0)

                # Predict the value using Random Forest
                forest_prediction = forest_reg.predict(new_data)[0]

                # Append the results to the list
                results.append({
                    'Company': input_Company.replace('Company_', ' '),
                    'Brand': input_Brand.replace('Brand_', ' '),
                    'Product': input_Product.replace('Product_', ' '),
                    'Month': input_Month.replace('Month_', ' '),
                    "Year": input_year,
                    'Prediction': forest_prediction
                })

    # Convert the results list to a dataframe
    df_results = pd.DataFrame(results)

    # Save the dataframe as a CSV file
    df_results.to_csv('results.csv', index=False)

    st.subheader("Prediction for new period :")
    # Print the dataframe in streamlit
    st.write(df_results)

    st.download_button(
        label="Download data as CSV",
        data=df_results.to_csv().encode('utf-8'),
        file_name='large_df.csv',
        mime='text/csv',
    )

    st.write(
        "How many Times Random Forest Predicted is Better than Budgeted", forest_pre)
    st.write("How many Times Budgeted is Better than Predicted", forest_bud)

    st.header("Thank you!")
except:
    pass

# killall streamlit
# nohup streamlit run app_final.py
