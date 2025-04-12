import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    fact_table = pd.read_excel("case-study-data.xlsx", sheet_name="Fact_table")
    trans_dim = pd.read_excel("case-study-data.xlsx", sheet_name="Trans_dim")
    item_dim = pd.read_excel("case-study-data.xlsx", sheet_name="Item_dim")
    customer_dim = pd.read_excel("case-study-data.xlsx", sheet_name="Customer_dim")
    time_dim = pd.read_excel("case-study-data.xlsx", sheet_name="Time_dim")
    store_dim = pd.read_excel("case-study-data.xlsx", sheet_name="Store_dim")

    # Merge datasets
    df = fact_table.merge(trans_dim, on='payment_key') \
                   .merge(item_dim, on='item_key') \
                   .merge(customer_dim, on='customer_key') \
                   .merge(time_dim, on='time_key') \
                   .merge(store_dim, on='store_key')

    for col in ['street', 'name', 'bank_name', 'unit_x']:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Aggregate data
    aggregated_data = df.groupby(['store_size', 'division_x']).agg(
        total_revenue=('total_price', 'sum'),
        total_quantity_sold=('quantity_sold', 'sum')
    ).reset_index()

    return aggregated_data

# Train model
def train_model(aggregated_data):
    X = aggregated_data[['store_size', 'division_x', 'total_quantity_sold']].copy()
    y = aggregated_data['total_revenue']

    # Encode categorical columns
    lb_encoders = {}
    for col in ['store_size', 'division_x']:
        encoder = LabelEncoder()
        X.loc[:, col] = encoder.fit_transform(X[col])
        lb_encoders[col] = encoder

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation")
    r2 = r2_score(y_test, y_pred)
    st.write(f"**R² Score:** {r2 * 100}")

    return model, lb_encoders

# Prediction logic
def make_prediction(model, lb_encoders, target_revenue):
    store_sizes = lb_encoders['store_size'].classes_
    divisions = lb_encoders['division_x'].classes_

    best_combination = None
    best_revenue_diff = float('inf')

    for store_size in store_sizes:
        for division in divisions:
            # Encode inputs
            store_encoded = lb_encoders['store_size'].transform([store_size])[0]
            division_encoded = lb_encoders['division_x'].transform([division])[0]
            total_quantity_sold = 10.0

            for _ in range(100):
                features = pd.DataFrame([{
                    'store_size': store_encoded,
                    'division_x': division_encoded,
                    'total_quantity_sold': total_quantity_sold
                }])

                predicted_revenue = model.predict(features)[0]
                revenue_diff = abs(target_revenue - predicted_revenue)

                if revenue_diff < 1:
                    break

                total_quantity_sold *= target_revenue / (predicted_revenue + 1e-6)

            if revenue_diff < best_revenue_diff:
                best_revenue_diff = revenue_diff
                best_combination = {
                    'store_size': store_size,
                    'division': division,
                    'total_quantity_sold': total_quantity_sold,
                    'predicted_revenue': predicted_revenue
                }

    return best_combination

# Streamlit UI
def main():
    st.set_page_config(page_title="Store Revenue Predictor", layout="wide")
    st.title("🔍 Store Revenue Prediction App")

    with st.spinner("Loading and training model..."):
        data = load_data()
        model, lb_encoders = train_model(data)

    st.header("🎯 Predict Store Combination for Target Revenue")
    target_revenue = st.number_input("Enter your target revenue:", min_value=0.0, value=1000.0, step=100.0)

    if st.button("Predict Best Combination"):
        with st.spinner("Running predictions..."):
            result = make_prediction(model, lb_encoders, target_revenue)

        if result:
            st.success("Optimal combination found!")
            st.markdown(f"**Store Size:** {result['store_size']}")
            st.markdown(f"**Division:** {result['division']}")
            st.markdown(f"**Total Quantity Sold:** {result['total_quantity_sold']:.2f}")
            st.markdown(f"**Predicted Revenue:** {result['predicted_revenue']:.2f}")
        else:
            st.error("No suitable combination found.")

if __name__ == "__main__":
    main()
