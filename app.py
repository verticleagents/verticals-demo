from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pandas as pd
import logging
import datetime

# Initialize Flask app
app = Flask(__name__)

# Add configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Optional: Limit file size to 16MB

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to the terminal
    ]
)

def perform_analysis(filepath):
    """
    Perform analysis and return metrics based on the uploaded Excel file.
    """
    logging.info("Performing analysis on the uploaded file.")
    
    excel_file = pd.ExcelFile(filepath)
    
    # Load sheets
    customers = excel_file.parse("Customers")
    transactions = excel_file.parse("Transactions")
    customer_feedback = excel_file.parse("Customer_Feedback")
    
    # High-value customer analysis
    high_value_customers = customers[customers['total_spend'] > customers['total_spend'].quantile(0.75)]
    high_value_customers_count = len(high_value_customers)

    # Log high-value customer results
    logging.info(f"High-value customers count: {high_value_customers_count}")

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

    # Log churned customers analysis
    logging.info(f"Churned customers count: {churned_customers_count}")
    logging.info(f"Churned customers by age group: {churned_by_age_group}")

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

    logging.info("Analysis completed successfully.")
    return {
        "metrics": {
            "high_value_customers_count": high_value_customers_count,
            "churned_customers_count": churned_customers_count,
            "churned_by_age_group": churned_by_age_group,
            "store_performance_summary": store_performance_summary,
            "average_ratings_by_store": avg_ratings_summary
        }
    }

@app.route('/')
def home():
    logging.info("Serving upload page.")
    return render_template("upload.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.warning("No file part in the request.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        logging.warning("No file selected for upload.")
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save the uploaded file with a consistent name
        filename = 'uploaded_file.xlsx'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"File uploaded successfully: {filepath}")
        return jsonify({
            "message": "File uploaded successfully. Click 'Begin Analysis' to start.", 
            "status": "success"
        })

@app.route('/begin_analysis', methods=['GET'])
def begin_analysis():
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.xlsx')
        
        # Check if file exists
        if not os.path.exists(filepath):
            logging.error("Analysis requested but no file found.")
            return jsonify({"error": "No file found for analysis"}), 404
            
        # Perform the analysis
        results = perform_analysis(filepath)
        
        return jsonify({
            "message": "Analysis completed successfully.",
            "results": results["metrics"]
        })
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logging.info("Starting Flask application.")
    app.run(debug=True, port=8080)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
