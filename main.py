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

# Load the Excel file
excel_file = pd.ExcelFile("data/complete_cafe_analytics.xlsx")  # Ensure the file path is correct

# Parse sheets
menu = excel_file.parse("Menu")
customers = excel_file.parse("Customers")
transactions = excel_file.parse("Transactions")
order_details = excel_file.parse("Order_Details")
customer_feedback = excel_file.parse("Customer_Feedback")
store_location = excel_file.parse("Store_Locations")
summary_statistics = excel_file.parse("Summary_Statistics")
customer_segments = excel_file.parse("Customer_Segments")
store_performance = excel_file.parse("Store_Performance")
product_analysis = excel_file.parse("Product_Analysis")
time_analysis = excel_file.parse("Time_Analysis")


def plot_to_base64(fig):
    """Convert Matplotlib plot to base64 for Flask response."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')


@app.route('/')
def home():
    return "Cafe Analytics Flask App is running!"


@app.route('/high_value_customers', methods=['GET'])
def high_value_customers():
    high_value = customers[customers['total_spend'] > customers['total_spend'].quantile(0.75)]
    high_value_sorted = high_value.sort_values(by='total_spend', ascending=False)
    return high_value_sorted.head(10).to_json(orient='records')


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


@app.route('/hourly_analysis', methods=['GET'])
def hourly_analysis():
    transactions['hour'] = transactions['date_time'].dt.hour
    hourly_analysis = transactions.groupby('hour').agg(
        transaction_count=('transaction_id', 'count'),
        avg_transaction_value=('final_total', 'mean'),
    ).reset_index()
    return hourly_analysis.to_json(orient='records')


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
    app.run(debug=True, port=5050)
