import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Define the directory where your data files are located, relative to the app.py script
# Adjust this path if your data files are in a different location
# EN LOCAL : DATA_DIR = 'C:\\Users\\jrmer\\OneDrive\\Bureau\\Coding\\Projets\\Rossmann_Sales_Forecast\\rossmann-store-sales\\'
DATA_DIR = 'rossmann-store-sales'

# Construct the full paths to the data files
model_path = os.path.join(DATA_DIR, 'refined_xgb_model.joblib')
train_store_transformed_df_path = os.path.join(DATA_DIR, 'train_store_transformed2.csv')
train_df_path = os.path.join(DATA_DIR, 'train.csv')
train_store_df_path_original = os.path.join(DATA_DIR, 'train_store.csv') # Use the merged data before sampling/optimization


# Load the trained model
try:
    refined_xgb_model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please ensure the model file is in the '{DATA_DIR}' directory.")
    st.stop()

# Load the original train_store_transformed2.csv to get necessary data for feature engineering
# This file contains information needed to recreate features like SalesDayOfWeekMean, AvgBasket, etc.
try:
    train_store_transformed_df_for_features = pd.read_csv(train_store_transformed_df_path)
    train_store_transformed_df_for_features['Date'] = pd.to_datetime(train_store_transformed_df_for_features['Date'])
except FileNotFoundError:
    st.error(f"Error: {train_store_transformed_df_path} not found. Please ensure the file is in the '{DATA_DIR}' directory.")
    st.stop()


# --- Feature Engineering Functions (recreating the logic from the notebook) ---

def calculate_sales_day_of_week_mean(df, original_train_df):
    """
    Calculates the mean sales per store and day of the week from the original training data
    and merges it with the input DataFrame.
    """
    # Ensure Date is datetime in original_train_df
    original_train_df['Date'] = pd.to_datetime(original_train_df['Date'])

    sales_day_of_week_mean = original_train_df.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()
    sales_day_of_week_mean.rename(columns={'Sales': 'SalesDayOfWeekMean_y'}, inplace=True) # Use _y to match the notebook

    df = pd.merge(df, sales_day_of_week_mean, on=['Store', 'DayOfWeek'], how='left')
    return df

def engineer_time_related_features(df):
    """
    Extracts time-related features from the 'Date' column.
    """
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfMonth'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter
    return df

def engineer_promotion_related_features(df):
    """
    Calculates 'Days since last promotion' and 'Number of consecutive promotional days'.
    Note: This requires sorting by store and date, which might be slow for large inputs.
    Consider pre-calculating this for all possible store/date combinations if feasible.
    For a single prediction, this logic needs to be adapted or precomputed data used.
    For simplicity in this example, we'll keep the structure but acknowledge it's not ideal for single row prediction.
    """
    # This logic is primarily for the full dataset. For a single row, it's complex.
    # A more robust deployment would pre-calculate these or use a different approach.
    # For demonstration, we'll keep the structure but acknowledge it's not ideal for single row prediction.
    df = df.sort_values(by=['Store', 'Date']) # Sorting is crucial for diff()

    # Calculate 'Days since last promotion'
    promo_changes = df.groupby('Store')['Promo'].diff().fillna(0)
    promo_starts = (promo_changes == 1).astype(int)
    df['DaysSinceLastPromo'] = df.groupby('Store')['Date'].diff().dt.days.where(promo_starts == 0, 0).fillna(0).cumsum()

    # Calculate 'Number of consecutive promotional days'
    consecutive_promo = df.groupby('Store')['Promo'].apply(lambda x: x.replace(0, pd.NA).groupby(x.notna().cumsum()).cumcount() + 1).fillna(0)
    df['ConsecutivePromoDays'] = consecutive_promo.reset_index(level=0, drop=True).astype(int)

    return df

def engineer_competition_related_features(df):
    """
    Applies log transformation to 'CompetitionDistance' and creates 'CompetitionOpen'.
    """
    df['CompetitionDistance_log'] = np.log1p(df['CompetitionDistance'])
    # Assuming that if CompetitionOpenSinceMonth/Year were not null in training, competition was open
    # For prediction, we might not have these columns directly. Need to infer or handle.
    # If the model was trained with 'CompetitionOpen' as a binary feature, we need to recreate it.
    # Based on the notebook, it was created from non-null CompetitionOpenSinceMonth.
    # If those columns were dropped, we need to ensure the 'CompetitionOpen' column is present.
    # Let's assume 'CompetitionOpen' is expected by the model and handle potential missingness.
    # In a real scenario, you'd need to know how 'CompetitionOpen' was determined.
    # For now, let's create a placeholder or ensure it's in the input data if needed by the model.
    # Based on the notebook, CompetitionOpenSinceMonth/Year were dropped later, but CompetitionOpen was engineered.
    # This implies CompetitionOpen was likely based on whether those original columns had values.
    # If the input data doesn't have those, we'd need an alternative way to determine CompetitionOpen.
    # Let's assume the input data *might* have these or we need a default.
    # A safer approach is to ensure the 'CompetitionOpen' column exists after data loading/initial processing.
    # Assuming for the app, we might not have these exact columns. Let's assume 'CompetitionOpen'
    # was a feature that needs to be present. If not in input, default to 1 or handle.
    # Recreating based on the notebook's logic:
    # This part is tricky without the original 'store.csv' data easily available in the app's input.
    # A practical solution for deployment is to use pre-calculated store features or
    # simplify the feature engineering. However, to match the trained model, we must
    # attempt to recreate the features as they were during training.
    # Let's assume 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' are NOT in the app input.
    # We need 'CompetitionOpen' to be present. If it's not engineered from other features in the app input,
    # we need to handle it. Let's assume the trained model expects it.
    # If the model was trained on a dataframe that had 'CompetitionOpen' as a column,
    # we need to ensure the input dataframe also has this column.
    # A simple way to ensure it exists for prediction if it was in training:
    if 'CompetitionOpen' not in df.columns:
         # This is a simplification. The true logic depends on the original feature engineering.
         # If the model relied on this being accurately derived from CompetitionOpenSinceMonth/Year,
         # this simple addition might not be sufficient.
         df['CompetitionOpen'] = 1 # Default to 1 (assuming most stores had competition open in the training period)


    return df

def engineer_store_specific_features(df, original_train_store_df):
    """
    Calculates store-level average sales and average customers and merges them.
    Requires the original merged train_store_df to calculate these averages.
    """
    # Ensure necessary columns are present for grouping
    if 'Store' not in original_train_store_df.columns or 'Sales' not in original_train_store_df.columns or 'Customers' not in original_train_store_df.columns:
         st.error("Error: Original training data missing 'Store', 'Sales', or 'Customers' columns for store-specific feature engineering.")
         st.stop()


    # Calculate store-level average sales
    store_avg_sales = original_train_store_df.groupby('Store')['Sales'].mean().reset_index()
    store_avg_sales.rename(columns={'Sales': 'StoreAvgSales'}, inplace=True)

    # Calculate store-level average customers
    store_avg_customers = original_train_store_df.groupby('Store')['Customers'].mean().reset_index()
    store_avg_customers.rename(columns={'Customers': 'StoreAvgCustomers'}, inplace=True)

    # Merge these new features back into the input DataFrame
    df = pd.merge(df, store_avg_sales, on='Store', how='left')
    df = pd.merge(df, store_avg_customers, on='Store', how='left')

    return df

def engineer_interaction_features(df):
    """
    Engineers interaction features between relevant columns.
    """
    # Engineer interaction features between 'Promo' and 'DayOfWeek'
    for i in range(1, 8):
        df[f'Promo_DayOfWeek_{i}'] = df['Promo'] * (df['DayOfWeek'] == i).astype(int)

    # Engineer interaction features between 'StoreType' and 'Promo'
    # Ensure StoreType columns exist after one-hot encoding
    for store_type in ['a', 'b', 'c', 'd']:
        if f'StoreType_{store_type}' in df.columns:
             df[f'StoreType_{store_type}_Promo'] = df[f'StoreType_{store_type}'] * df['Promo']
        else:
             # Handle cases where a StoreType might not be present in the input data
             # For a single prediction, this might happen. Add the column with 0s.
             df[f'StoreType_{store_type}_Promo'] = 0


    # Engineer interaction features between 'Assortment' and 'Promo'
    # Ensure Assortment columns exist after one-hot encoding
    for assortment_type in ['a', 'b', 'c']:
        if f'Assortment_{assortment_type}' in df.columns:
            df[f'Assortment_{assortment_type}_Promo'] = df[f'Assortment_{assortment_type}'] * df['Promo']
        else:
            # Handle cases where an Assortment might not be present in the input data
            # For a single prediction, this might happen. Add the column with 0s.
            df[f'Assortment_{assortment_type}_Promo'] = 0


    # Example: Interaction between SalesDayOfWeekMean and Promo
    if 'SalesDayOfWeekMean_y' in df.columns:
        df['SalesDayOfWeekMean_Promo_Interaction'] = df['SalesDayOfWeekMean_y'] * df['Promo']
    else:
        # Handle if SalesDayOfWeekMean_y was not successfully merged
        df['SalesDayOfWeekMean_Promo_Interaction'] = 0


    return df

def create_binned_competition_distance(df, original_competition_distance_series):
    """
    Creates binned features for CompetitionDistance based on the original data's distribution.
    Requires the original CompetitionDistance data to define the bins.
    """
    # Define bins based on the original training data's quantiles
    # Ensure original_competition_distance_series is a pandas Series
    if not isinstance(original_competition_distance_series, pd.Series):
         st.error("Error: Original CompetitionDistance data is not a pandas Series.")
         st.stop()

    # Handle potential NaNs in the original series before calculating quantiles
    original_competition_distance_series = original_competition_distance_series.dropna()


    try:
        # Use the same quantile strategy as in the notebook
        df['CompetitionDistance_Bin'] = pd.qcut(df['CompetitionDistance'], q=5, labels=False, duplicates='drop')
    except Exception as e:
        st.warning(f"Could not create CompetitionDistance_Bin. Ensure CompetitionDistance is numeric and has enough unique values. Error: {e}")
        # If binning fails, create the column with a default value or handle appropriately
        df['CompetitionDistance_Bin'] = -1 # Or some other indicator of failure/missing bin


    return df


def prepare_input_features(input_df, original_train_df_for_features, original_train_store_df_for_features):
    """
    Prepares the input DataFrame for prediction by applying the same feature engineering
    steps as used during model training.
    """
    df = input_df.copy()

    # Ensure 'Date' is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("Error: 'Date' column not found in input data.")
        st.stop()

    # Ensure categorical columns are handled (one-hot encoded if necessary)
    # Assuming 'StoreType', 'Assortment', 'PromoInterval', 'StateHoliday' were one-hot encoded
    # Need to re-apply one-hot encoding to the input data
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday'], dummy_na=False)

    # Rename PromoInterval columns to match training if necessary
    df.rename(columns={
        'PromoInterval_Jan,Apr,Jul,Oct': 'PromoInterval_JanAprJulOct',
        'PromoInterval_Feb,May,Aug,Nov': 'PromoInterval_FebMayAugNov',
        'PromoInterval_Mar,Jun,Sept,Dec': 'PromoInterval_MarJunSeptDec'
    }, inplace=True)


    # Engineer time-related features
    df = engineer_time_related_features(df)

    # Calculate and merge 'Sales Day Of The Week Mean'
    df = calculate_sales_day_of_week_mean(df, original_train_df_for_features.copy()) # Use a copy to avoid modifying the original

    # Engineer promotion-related features (simplified for single row prediction)
    # This needs careful consideration for a single row. The current notebook logic is for a time series.
    # For a single prediction, you likely need pre-calculated data for DaysSinceLastPromo and ConsecutivePromoDays.
    # As a placeholder, let's assume they are 0 or some default if not in input.
    # In a real deployment, you'd need a lookup table or a more sophisticated approach.
    if 'DaysSinceLastPromo' not in df.columns:
        df['DaysSinceLastPromo'] = 0 # Placeholder
    if 'ConsecutivePromoDays' not in df.columns:
        df['ConsecutivePromoDays'] = 0 # Placeholder


    # Engineer competition-related features
    df = engineer_competition_related_features(df)

    # Engineer store-specific features
    df = engineer_store_specific_features(df, original_train_store_df_for_features.copy()) # Use a copy

    # Engineer interaction features
    df = engineer_interaction_features(df)

    # Create binned CompetitionDistance
    # Need the original CompetitionDistance series from the training data
    if 'CompetitionDistance' in original_train_store_df_for_features.columns:
        df = create_binned_competition_distance(df, original_train_store_df_for_features['CompetitionDistance'])
    else:
         st.warning("Original CompetitionDistance not available for binning.")
         # If original data not available, create the bin column with a default
         df['CompetitionDistance_Bin'] = -1


    # Handle AvgBasket - re-calculate for the input row if Customers > 0
    if 'Sales' in df.columns and 'Customers' in df.columns:
         df['AvgBasket'] = df.apply(
            lambda row: row['Sales'] / row['Customers'] if row['Customers'] > 0 else np.nan, axis=1
         )
         # Impute any resulting NaNs (shouldn't be many if Customers > 0 check is done)
         if df['AvgBasket'].isnull().sum() > 0:
             # Use a fallback mean if original_train_store_df_for_features is available
             if 'AvgBasket' in original_train_store_df_for_features.columns:
                  mean_avg_basket = original_train_store_df_for_features['AvgBasket'].mean()
                  df['AvgBasket'].fillna(mean_avg_basket, inplace=True)
             else:
                  df['AvgBasket'].fillna(0, inplace=True) # Fallback to 0 if mean not available
    else:
         # If Sales or Customers columns are missing in input, add AvgBasket as NaN or 0
         df['AvgBasket'] = np.nan # Or 0 if preferred


    # --- Align columns with the model's expected features ---
    # Get the list of features the model was trained on
    # Assuming the model object has a .feature_names_in_ attribute or similar
    # If not, you need to manually get the feature order from your training script
    if hasattr(refined_xgb_model, 'feature_names_in_'):
        model_features_order = refined_xgb_model.feature_names_in_
    else:
        # Fallback: Try to infer from the shape or rely on a predefined list
        # This is less reliable. It's best to save the feature names during training.
        # For this example, let's assume we know the expected columns based on the notebook's final dataframe
        # This list must EXACTLY match the columns used to train refined_xgb_model
        model_features_order = [
            'Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'SchoolHoliday',
            'AvgBasket', 'CompetitionDistance', 'Promo2', 'DaysSinceLastPromo',
            'ConsecutivePromoDays', 'CompetitionDistance_log', 'CompetitionOpen',
            'StoreAvgSales', 'StoreAvgCustomers', 'DayOfYear', 'WeekOfYear',
            'DayOfMonth', 'Quarter', 'SalesDayOfWeekMean_y',
            'PromoInterval_FebMayAugNov', 'PromoInterval_JanAprJulOct',
            'PromoInterval_MarJunSeptDec', 'StoreType_a', 'StoreType_b',
            'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b',
            'Assortment_c', 'Promo_DayOfWeek_1', 'Promo_DayOfWeek_2',
            'Promo_DayOfWeek_3', 'Promo_DayOfWeek_4', 'Promo_DayOfWeek_5',
            'Promo_DayOfWeek_6', 'Promo_DayOfWeek_7', 'StoreType_a_Promo',
            'StoreType_b_Promo', 'StoreType_c_Promo', 'StoreType_d_Promo',
            'Assortment_a_Promo', 'Assortment_b_Promo', 'Assortment_c_Promo',
            'SalesDayOfWeekMean_Promo_Interaction', 'CompetitionDistance_Bin',
            # Add any other features the model was trained on
        ]
        # Ensure all expected columns are in the dataframe, add missing ones with default values (e.g., 0)
        for col in model_features_order:
            if col not in df.columns:
                st.warning(f"Adding missing column: {col}")
                df[col] = 0 # Default value for missing features


    # Reindex the input DataFrame to match the model's feature order
    # This is CRUCIAL for preventing feature mismatch errors
    try:
        df = df.reindex(columns=model_features_order, fill_value=0) # Fill missing with 0
    except Exception as e:
        st.error(f"Error reindexing columns: {e}")
        st.stop()


    # Ensure numerical columns are scaled using the same scaler fitted during training
    # This requires saving and loading the scaler as well
    # For this example, let's skip scaling in the app for simplicity, assuming the model
    # can handle unscaled data or was trained on data where scaling was not critical
    # for the final set of features used by the refined XGBoost model.
    # If scaling was used, you MUST load and apply the scaler here.
    # scaler_path = os.path.join(DATA_DIR, 'scaler.joblib') # Assuming scaler is also saved in DATA_DIR
    # try:
    #     scaler = joblib.load(scaler_path)
    #     numerical_cols_to_scale = [col for col in model_features_order if df[col].dtype in ['int64', 'float64'] and col not in ['Store', 'DayOfWeek']]
    #     df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale])
    # except FileNotFoundError:
    #     st.warning(f"Scaler file not found at {scaler_path}. Skipping scaling.")
    # except Exception as e:
    #      st.warning(f"Error applying scaler: {e}. Skipping scaling.")


    return df


# --- Streamlit App Interface ---

st.title('Rossmann Store Sales Prediction')

st.sidebar.header('Input Features')

# Collect input features from the user
store = st.sidebar.slider('Store', 1, 1115, 1)
dayofweek = st.sidebar.slider('DayOfWeek', 1, 7, 1)
date = st.sidebar.date_input('Date')
customers = st.sidebar.number_input('Customers', 0, 10000, 500)
promo = st.sidebar.selectbox('Promo', [0, 1])
stateholiday = st.sidebar.selectbox('StateHoliday', ['0', 'a', 'b', 'c']) # Use strings to match data
schoolholiday = st.sidebar.selectbox('SchoolHoliday', [0, 1])
# AvgBasket will be calculated
competitiondistance = st.sidebar.number_input('CompetitionDistance', 0.0, 100000.0, 1000.0)
# CompetitionOpenSinceMonth/Year are not directly used as input based on the notebook's final features
promo2 = st.sidebar.selectbox('Promo2', [0, 1])
# Promo2SinceWeek/Year are not directly used as input based on the notebook's final features
promointerval = st.sidebar.selectbox('PromoInterval', ['nan', 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec']) # Include 'nan' for missing
storetype = st.sidebar.selectbox('StoreType', ['a', 'b', 'c', 'd'])
assortment = st.sidebar.selectbox('Assortment', ['a', 'b', 'c'])
# Other engineered features will be calculated

# Create a DataFrame from user input
input_data = {
    'Store': store,
    'DayOfWeek': dayofweek,
    'Date': pd.to_datetime(date), # Convert date input to datetime
    'Customers': customers,
    'Open': 1, # Assuming predicting for open stores
    'Promo': promo,
    'StateHoliday': stateholiday,
    'SchoolHoliday': schoolholiday,
    'CompetitionDistance': competitiondistance,
    'Promo2': promo2,
    'PromoInterval': promointerval if promointerval != 'nan' else None, # Handle 'nan' for missing
    'StoreType': storetype,
    'Assortment': assortment,
    # Add placeholders for columns needed for feature engineering if not directly from input
    'Sales': 0, # Placeholder for AvgBasket calculation - will be overwritten
    # Add placeholder columns that might have been used in original feature engineering
    # These columns were used to create CompetitionOpen and other features.
    # While the engineered features are directly used by the model, the original columns
    # are needed *within* prepare_input_features to recreate those engineered features
    # if the logic relies on them (e.g., CompetitionOpen from non-null checks).
    # However, the final trained model dropped these. The prepare_input_features needs
    # to be self-contained or use precalculated values if original columns are not available.
    # The current prepare_input_features assumes access to original dataframes for some steps.
    # If running locally without original dataframes, these feature engineering steps need
    # to be modified to use precalculated store-specific averages etc.
    # For now, let's keep the structure assuming access to original dataframes for feature calc.
    'CompetitionOpenSinceMonth': None, # Placeholder - not used by the final model, but might be in intermediate steps
    'CompetitionOpenSinceYear': None, # Placeholder - not used by the final model, but might be in intermediate steps
    'Promo2SinceWeek': None, # Placeholder - not used by the final model, but might be in intermediate steps
    'Promo2SinceYear': None, # Placeholder - not used by the final model, but might be in intermediate steps
}

input_df = pd.DataFrame([input_data])

# Ensure 'Open' is 1 for prediction as the model was trained on open stores
input_df = input_df[input_df['Open'] == 1].copy()


if st.sidebar.button('Predict Sales'):
    # Prepare the input features
    # Need the original training dataframes to calculate features like SalesDayOfWeekMean, StoreAvgSales/Customers
    # Assuming train_df and train_store_df are needed for these calculations
    # Load them here or ensure they are accessible
    try:
        original_train_df_for_features = pd.read_csv(train_df_path)
        original_train_store_df_for_features = pd.read_csv(train_store_df_path_original)
    except FileNotFoundError:
        st.error("Error: Original training data files (train.csv or train_store.csv) not found in the '{DATA_DIR}' directory.")
        st.stop()


    processed_input = prepare_input_features(
        input_df,
        original_train_df_for_features.copy(), # Pass copies to avoid modification
        original_train_store_df_for_features.copy()
    )

    # Ensure the processed input DataFrame has the same columns as the model expects, in the same order
    # This is a critical step to prevent the Feature names mismatch error
    if hasattr(refined_xgb_model, 'feature_names_in_'):
        model_features_order = refined_xgb_model.feature_names_in_
        # Reindex again just to be absolutely sure
        try:
            processed_input = processed_input.reindex(columns=model_features_order, fill_value=0)
        except Exception as e:
            st.error(f"Final reindexing failed: {e}")
            st.stop()

        # Ensure dtypes match if necessary (XGBoost is usually flexible, but can be an issue)
        # You might need to convert columns to numeric types expected by the model if they are objects/booleans
        for col in processed_input.columns:
             if processed_input[col].dtype == 'bool':
                 processed_input[col] = processed_input[col].astype(int) # Convert booleans to int (0 or 1)
             # Add other dtype conversions if needed


        # Make prediction
        try:
            prediction = refined_xgb_model.predict(processed_input)
            st.subheader('Predicted Sales')
            st.write(f'The predicted sales are: {prediction[0]:,.2f}')
        except Exception as e:
             st.error(f"Error during prediction: {e}")
             st.write("Processed Input DataFrame columns and order:")
             st.write(processed_input.columns.tolist())
             st.write("Model Expected Features:")
             st.write(model_features_order.tolist())
             st.write("Processed Input DataFrame head:")
             st.write(processed_input.head())


    else:
        st.error("Could not get model feature names. Cannot ensure correct input features.")
        st.write("Processed Input DataFrame columns:")
        st.write(processed_input.columns.tolist())