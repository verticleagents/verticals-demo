import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from werkzeug.utils import secure_filename

# Create directories for uploads and static figures
UPLOAD_FOLDER = './uploads'
FIGURE_FOLDER = './static/figures'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)

# Flask App Configuration
app = Flask(__name__)
# Configure upload folder and maximum file size
app.config['UPLOAD_FOLDER'] = './uploads'  # Ensure this folder exists or create it
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    API to upload an Excel file and trigger the analysis.
    """

    # Check if the 'file' field is in the request
    if 'file' not in request.files:
        app.logger.error("File field is missing in the request.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        app.logger.error("No file was selected for upload.")
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    if not file.filename.endswith('.xlsx'):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Only .xlsx files are allowed"}), 400

    # Save the file securely
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.info(f"File saved successfully at {filepath}")

        # Perform analysis on the uploaded file
        try:
            analysis_results = perform_analysis(filepath)
            return jsonify(analysis_results)
        except Exception as e:
            app.logger.error(f"Error during analysis: {str(e)}")
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"Error saving file: {str(e)}")
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

# def perform_analysis(filepath):
#     """
#     Perform analysis and save figures based on the uploaded Excel file.
#     """
#     excel_file = pd.ExcelFile(filepath)
    
#     # Load sheets
#     menu = excel_file.parse("Menu")
#     customers = excel_file.parse("Customers")
#     transactions = excel_file.parse("Transactions")
#     order_details = excel_file.parse("Order_Details")
#     customer_feedback = excel_file.parse("Customer_Feedback")
#     store_location = excel_file.parse("Store_Locations")

#     # High-value customer analysis
#     high_value_customers = customers[customers['total_spend'] > customers['total_spend'].quantile(0.75)]
#     high_value_customers = high_value_customers.sort_values(by='total_spend', ascending=False)
#     top_10_high_value = high_value_customers.head(10)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=top_10_high_value, x='customer_id', y='total_spend', palette='viridis', ax=ax)
#     ax.set_title("Top 10 High-Value Customers by Total Spend")
#     figpath1 = os.path.join(app.config['FIGURE_FOLDER'], 'top_10_high_value_customers.png')
#     plt.savefig(figpath1)
#     plt.close()

#     # Churned customers analysis
#     transactions['date_time'] = pd.to_datetime(transactions['date_time'])
#     customers['join_date'] = pd.to_datetime(customers['join_date'])
#     latest_transactions = transactions.groupby('customer_id')['date_time'].max().reset_index()
#     latest_transactions.rename(columns={'date_time': 'last_transaction_date'}, inplace=True)
#     customers_with_last_transaction = customers.merge(latest_transactions, on='customer_id', how='left')
#     six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
#     churned_customers = customers_with_last_transaction[
#         (customers_with_last_transaction['last_transaction_date'] < six_months_ago) |
#         (customers_with_last_transaction['last_transaction_date'].isna())
#     ]
#     churned_by_age_group = churned_customers.groupby('age_group').size().reset_index(name='count')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=churned_by_age_group, x='age_group', y='count', palette='coolwarm', ax=ax)
#     ax.set_title("Churned Customers by Age Group")
#     figpath2 = os.path.join(app.config['FIGURE_FOLDER'], 'churned_customers_by_age_group.png')
#     plt.savefig(figpath2)
#     plt.close()

#     # Aggregate store performance
#     store_performance = transactions.groupby('store_location').agg(
#         total_transactions=('transaction_id', 'count'),
#         avg_transaction_value=('final_total', 'mean'),
#     ).reset_index()
#     store_ratings = customer_feedback.groupby('transaction_id')['rating_overall'].mean().reset_index()
#     store_transactions = transactions.merge(store_ratings, on='transaction_id', how='left')
#     store_avg_ratings = store_transactions.groupby('store_location')['rating_overall'].mean().reset_index()
#     store_performance = store_performance.merge(store_avg_ratings, on='store_location', how='left')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=store_performance, x='store_location', y='total_transactions', palette='magma', ax=ax)
#     ax.set_title("Total Transactions by Store Location")
#     figpath3 = os.path.join(app.config['FIGURE_FOLDER'], 'total_transactions_by_store.png')
#     plt.savefig(figpath3)
#     plt.close()

#     return {
#         "figures": {
#             "top_10_high_value_customers": figpath1,
#             "churned_customers_by_age_group": figpath2,
#             "total_transactions_by_store": figpath3
#         },
#         "metrics": {
#             "high_value_customers_count": len(high_value_customers),
#             "churned_customers_count": len(churned_customers),
#             "store_performance_summary": store_performance.to_dict(orient='records')
#         }
#     }


def perform_analysis(filepath):
    """
    Perform analysis and return metrics based on the uploaded Excel file.
    """
    excel_file = pd.ExcelFile(filepath)
    
    # Load sheets
    customers = excel_file.parse("Customers")
    transactions = excel_file.parse("Transactions")
    customer_feedback = excel_file.parse("Customer_Feedback")

    # High-value customer analysis
    high_value_customers = customers[customers['total_spend'] > customers['total_spend'].quantile(0.75)]
    high_value_customers_count = len(high_value_customers)

    # Churned customers analysis
    transactions['date_time'] = pd.to_datetime(transactions['date_time'])
    customers['join_date'] = pd.to_datetime(customers['join_date'])
    latest_transactions = transactions.groupby('customer_id')['date_time'].max().reset_index()
    latest_transactions.rename(columns={'date_time': 'last_transaction_date'}, inplace=True)
    customers_with_last_transaction = customers.merge(latest_transactions, on='customer_id', how='left')
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    churned_customers = customers_with_last_transaction[
        (customers_with_last_transaction['last_transaction_date'] < six_months_ago) |
        (customers_with_last_transaction['last_transaction_date'].isna())
    ]
    churned_customers_count = len(churned_customers)
    churned_by_age_group = churned_customers['age_group'].value_counts().to_dict()

    # Aggregate store performance
    store_performance = transactions.groupby('store_location').agg(
        total_transactions=('transaction_id', 'count'),
        avg_transaction_value=('final_total', 'mean'),
    ).reset_index()
    store_performance_summary = store_performance.to_dict(orient='records')

    # Feedback summary
    avg_ratings_by_store = customer_feedback.groupby('transaction_id')['rating_overall'].mean().reset_index()
    store_ratings = transactions.merge(avg_ratings_by_store, on='transaction_id', how='left')
    avg_ratings_summary = store_ratings.groupby('store_location')['rating_overall'].mean().to_dict()

    return {
        "metrics": {
            "high_value_customers_count": high_value_customers_count,
            "churned_customers_count": churned_customers_count,
            "churned_by_age_group": churned_by_age_group,
            "store_performance_summary": store_performance_summary,
            "average_ratings_by_store": avg_ratings_summary
        }
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
