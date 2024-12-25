from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import io
import base64

app = Flask(__name__)

# Utility function to convert plots to Base64
def plot_to_base64(fig):
    """Convert Matplotlib plot to base64 for Flask response."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

# Endpoint for receiving the Excel file dynamically
@app.route('/upload-file', methods=['POST'])
def upload_file():
    try:
        # Receive the file and JSON from the request
        file = request.files['file']
        sheets_json = request.form.get('sheets')  # JSON of sheets provided in the request
        
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Read Excel file
        excel_file = pd.ExcelFile(file)
        
        # Parse sheets based on provided JSON
        sheets = {}
        for sheet in pd.read_json(sheets_json).get("value", []):
            display_name = sheet['DisplayName']
            sheets[display_name] = excel_file.parse(display_name)
        
        # Store parsed sheets globally for analysis
        global menu, customers, transactions, order_details, customer_feedback
        global store_location, summary_statistics, customer_segments
        global store_performance, product_analysis, time_analysis

        menu = sheets.get("Menu")
        customers = sheets.get("Customers")
        transactions = sheets.get("Transactions")
        order_details = sheets.get("Order_Details")
        customer_feedback = sheets.get("Customer_Feedback")
        store_location = sheets.get("Store_Locations")
        summary_statistics = sheets.get("Summary_Statistics")
        customer_segments = sheets.get("Customer_Segments")
        store_performance = sheets.get("Store_Performance")
        product_analysis = sheets.get("Product_Analysis")
        time_analysis = sheets.get("Time_Analysis")

        return jsonify({"message": "File and sheets processed successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint for high-value customers
@app.route('/high_value_customers', methods=['GET'])
def high_value_customers():
    high_value = customers[customers['total_spend'] > customers['total_spend'].quantile(0.75)]
    high_value_sorted = high_value.sort_values(by='total_spend', ascending=False)
    return high_value_sorted.head(10).to_json(orient='records')

# Endpoint for churned customers
@app.route('/churned_customers', methods=['GET'])
def churned_customers():
    transactions['date_time'] = pd.to_datetime(transactions['date_time'])
    customers['join_date'] = pd.to_datetime(customers['join_date'])
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    latest_transactions = transactions.groupby('customer_id')['date_time'].max().reset_index()
    latest_transactions.rename(columns={'date_time': 'last_transaction_date'}, inplace=True)
    customers_with_last_transaction = customers.merge(latest_transactions, on='customer_id', how='left')
    churned = customers_with_last_transaction[
        (customers_with_last_transaction['last_transaction_date'] < six_months_ago) |
        (customers_with_last_transaction['last_transaction_date'].isna())
    ]
    return churned[['customer_id', 'age_group', 'total_spend']].to_json(orient='records')

# Endpoint for store performance metrics
@app.route('/store_performance', methods=['GET'])
def store_performance_metrics():
    store_performance = transactions.groupby('store_location').agg(
        total_transactions=('transaction_id', 'count'),
        avg_transaction_value=('final_total', 'mean'),
    ).reset_index()
    store_ratings = customer_feedback.groupby('transaction_id')['rating_overall'].mean().reset_index()
    store_transactions = transactions.merge(store_ratings, on='transaction_id', how='left')
    store_avg_ratings = store_transactions.groupby('store_location')['rating_overall'].mean().reset_index()
    store_performance = store_performance.merge(store_avg_ratings, on='store_location', how='left')
    return store_performance.to_json(orient='records')

# Endpoint for top products
@app.route('/top_products', methods=['GET'])
def top_products():
    product_performance = order_details.groupby('item_name').agg(
        total_orders=('quantity', 'sum'),
        total_revenue=('total_price', 'sum'),
    ).reset_index()
    product_ratings = customer_feedback.merge(order_details, on='transaction_id', how='left').groupby('item_name')['rating_overall'].mean().reset_index()
    product_performance = product_performance.merge(product_ratings, on='item_name', how='left')
    top_products = product_performance.sort_values(by='total_orders', ascending=False).head(10)
    return top_products.to_json(orient='records')

# Endpoint for hourly trend plot
@app.route('/hourly_trend_plot', methods=['GET'])
def hourly_trend_plot():
    transactions['hour'] = transactions['date_time'].dt.hour
    hourly_analysis = transactions.groupby('hour').agg(
        transaction_count=('transaction_id', 'count')
    ).reset_index()
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=hourly_analysis, x='hour', y='transaction_count')
    plt.title("Hourly Transaction Trends")
    plt.xlabel("Hour of Day")
    plt.ylabel("Transaction Count")
    image_base64 = plot_to_base64(fig)
    return jsonify({"plot": image_base64})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
