import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").strip().rstrip("/")
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")
st.title("SuperKart Product Store Sales Prediction")
st.caption("Inputs are validated against the categories and numeric ranges used during training.")

PERISHABLE_ITEMS = {
    "Fruits and Vegetables", "Dairy", "Meat", "Seafood", "Breads", "Frozen Foods"
}
PRODUCT_TYPES = [
    "Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
    "Health and Hygiene", "Baking Goods", "Breads", "Breakfast", "Frozen Foods",
    "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"
]
TRAINED_STORE_TYPES = [
    "Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"
]

st.subheader("Online prediction")
with st.form("online_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        Product_Weight = st.number_input("Product Weight", min_value=4.0, max_value=22.0, value=12.66, step=0.01)
        Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.004, max_value=0.298, value=0.068, step=0.001, format="%.3f")
        Product_MRP = st.number_input("Product MRP", min_value=31.0, max_value=266.0, value=147.03, step=0.01)
        Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        Product_Id_char = st.selectbox("Product ID Category", ["FD", "NC", "DR"])
    with col2:
        Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
        Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
        Store_Type = st.selectbox("Store Type", TRAINED_STORE_TYPES, index=2)
        # 🔴 CHANGE: age range matches the training data reference year (2026).
        Store_Age_Years = st.number_input("Store Age in 2026 (years)", min_value=17, max_value=39, value=17, step=1)
        Product_Type = st.selectbox("Product Type", PRODUCT_TYPES)

    Product_Type_Category = "Perishable" if Product_Type in PERISHABLE_ITEMS else "Non-Perishable"
    st.caption(f"Derived Product Type Category: **{Product_Type_Category}**")
    predict_clicked = st.form_submit_button("Predict sales", type="primary")

if predict_clicked:
    input_payload = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": int(Store_Age_Years),
        "Product_Type_Category": Product_Type_Category,
    }
    st.dataframe(pd.DataFrame([input_payload]), use_container_width=True)
    try:
        response = requests.post(f"{BACKEND_URL}/v1/superkart", json=input_payload, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            st.success(f"Predicted Product Store Sales: {result['Predicted Product Store Sales']:.2f}")
            if result.get("prediction_clipped"):
                st.warning("The model produced a negative raw estimate; it was clipped to zero. Check logs and model/data versions.")
        else:
            st.error(f"Backend returned HTTP {response.status_code}: {response.text[:2000]}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {BACKEND_URL}. Details: {exc}")
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unexpected API response: {exc}")

st.divider()
st.subheader("Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    st.caption("Required columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category.")
    if st.button("Predict batch", type="primary"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{BACKEND_URL}/v1/superkartbatch", files=files, timeout=120)
            if response.ok:
                result = response.json()
                result_df = pd.DataFrame({"Predicted Product Store Sales": result["predictions"]})
                st.success(f"Batch prediction completed for {result.get('count', len(result_df))} rows.")
                if result.get("clipped_negative_predictions", 0):
                    st.warning(f"{result['clipped_negative_predictions']} negative raw estimate(s) were clipped to zero; investigate model/data drift.")
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Backend returned HTTP {response.status_code}: {response.text[:3000]}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}. Details: {exc}")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Unexpected API response: {exc}")
