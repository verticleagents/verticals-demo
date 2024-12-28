from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import logging
import datetime
import json
from itertools import combinations
import numpy as np
# Initialize Flask app
app = Flask(__name__)

# Add configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()]
)

def analyze_customers(Customers, Transactions):
    logging.info("Starting customer analysis")

    # High value customers analysis
    high_value_threshold = Customers['total_spend'].quantile(0.75)
    high_value_customers = Customers[Customers['total_spend'] > high_value_threshold]
    high_value_customers = high_value_customers.sort_values(by='total_spend', ascending=False)
    
    # Churn analysis
    Transactions['date_time'] = pd.to_datetime(Transactions['date_time'])
    Customers['join_date'] = pd.to_datetime(Customers['join_date'])
    latest_transactions = Transactions.groupby('customer_id')['date_time'].max().reset_index()
    latest_transactions.rename(columns={'date_time': 'last_transaction_date'}, inplace=True)
    customers_with_last_transaction = Customers.merge(latest_transactions, on='customer_id', how='left')
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    churned_customers = customers_with_last_transaction[
        (customers_with_last_transaction['last_transaction_date'] < six_months_ago) |
        (customers_with_last_transaction['last_transaction_date'].isna())
    ]

    # Average spend by age group
    avg_spend_by_age = Customers.groupby('age_group')['total_spend'].mean().to_dict()

    # Average spend by visit frequency
    avg_spend_by_frequency = Customers.groupby('visit_frequency')['total_spend'].mean().to_dict()

    # Top 10 customers
    top_10_customers = high_value_customers.head(10)[['customer_id', 'total_spend']].to_dict('records')

    logging.info("Customer analysis completed")
    return {
        'high_value_customers_count': len(high_value_customers),
        'churned_customers_count': len(churned_customers),
        'avg_spend_by_age': avg_spend_by_age,
        'avg_spend_by_frequency': avg_spend_by_frequency,
        'top_customers': top_10_customers
    }



def analyze_store_performance(Transactions, Customer_Feedback):
    logging.info("Starting store performance analysis")

    # Store performance metrics
    store_performance = Transactions.groupby('store_location').agg(
        total_transactions=('transaction_id', 'count'),
        avg_transaction_value=('final_total', 'mean')
    ).reset_index()

    # Store ratings
    store_ratings = Customer_Feedback.groupby('transaction_id')['rating_overall'].mean().reset_index()
    store_transactions = Transactions.merge(store_ratings, on='transaction_id', how='left')
    store_avg_ratings = store_transactions.groupby('store_location')['rating_overall'].mean().reset_index()

    # Merge performance and ratings
    store_performance = store_performance.merge(store_avg_ratings, on='store_location', how='left')

    logging.info("Store performance analysis completed")
    return {
        'store_performance_summary': store_performance.to_dict('records'),
        'average_ratings_by_store': store_avg_ratings.set_index('store_location')['rating_overall'].to_dict()
    }

def analyze_products(Order_Details, Customer_Feedback, Transactions):
    logging.info("Starting product analysis")

    # Product performance metrics
    product_performance = Order_Details.groupby('item_name').agg(
        total_orders=('quantity', 'sum'),
        total_revenue=('total_price', 'sum')
    ).reset_index()

    # Product ratings
    feedback_with_products = Customer_Feedback.merge(
        Order_Details[['transaction_id', 'item_name']].drop_duplicates(),
        on='transaction_id',
        how='left'
    )
    product_ratings = feedback_with_products.groupby('item_name')['rating_overall'].mean().reset_index()

    # Merge performance and ratings
    product_performance = product_performance.merge(product_ratings, on='item_name', how='left')

    # Get top 10 products
    top_10_products = product_performance.sort_values(by='total_orders', ascending=False).head(10)

    logging.info("Product analysis completed")
    return {
        'top_products': top_10_products.to_dict('records'),
        'product_ratings': product_performance[['item_name', 'total_revenue', 'rating_overall']].to_dict('records')
    }
def perform_rfm_analysis(Customers, Transactions):
    """Perform RFM (Recency, Frequency, Monetary) analysis"""
    current_date = pd.Timestamp.now()
    
    # Calculate RFM metrics
    rfm = Transactions.groupby('customer_id').agg({
        'date_time': lambda x: (current_date - x.max()).days,  # Recency
        'transaction_id': 'count',  # Frequency
        'final_total': 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Create scores
    rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'], q=5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1,2,3,4,5])
    
    # Calculate RFM Score
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    # Segment customers
    def segment_customers(row):
        if row['r_score'] == 5 and row['f_score'] == 5:
            return 'Champions'
        elif row['r_score'] >= 4 and row['f_score'] >= 4:
            return 'Loyal Customers'
        elif row['r_score'] >= 3:
            return 'Active Customers'
        elif row['r_score'] == 1:
            return 'Lost Customers'
        else:
            return 'At Risk'
    
    rfm['customer_segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

def analyze_product_affinity(Order_Details):
    """Analyze which products are commonly purchased together"""
    from itertools import combinations
    
    # Get transactions with multiple items
    transaction_items = Order_Details.groupby('transaction_id')['item_name'].agg(list).reset_index()
    
    # Find frequent item pairs
    item_pairs = []
    support_dict = {}
    
    for items in transaction_items['item_name']:
        pairs = list(combinations(sorted(items), 2))
        item_pairs.extend(pairs)
        
    for pair in item_pairs:
        support_dict[pair] = support_dict.get(pair, 0) + 1
        
    # Convert to DataFrame
    affinity_df = pd.DataFrame(list(support_dict.items()), columns=['item_pair', 'frequency'])
    affinity_df['support'] = affinity_df['frequency'] / len(transaction_items)
    
    return affinity_df.sort_values('frequency', ascending=False).head(10)

def analyze_store_efficiency(Transactions, store_sizes, default_size=1000):
    """Calculate store efficiency metrics with fallback for missing store sizes"""
    store_metrics = Transactions.groupby('store_location').agg({
        'final_total': 'sum',
        'transaction_id': 'count'
    }).reset_index()

    store_sizes_series = pd.Series(store_sizes)
    store_metrics['store_size'] = store_metrics['store_location'].map(store_sizes_series).fillna(default_size)

    store_metrics['sales_per_sqft'] = store_metrics['final_total'] / store_metrics['store_size']
    store_metrics['avg_transaction_value'] = store_metrics['final_total'] / store_metrics['transaction_id']

    return store_metrics


def calculate_price_elasticity(Order_Details):
    """Calculate price elasticity of products"""
    product_metrics = Order_Details.groupby('item_name').agg({
        'unit_price': 'mean',
        'quantity': 'sum',
        'total_price': 'sum'
    }).reset_index()

    # Calculate price elasticity (handle NaN or division issues)
    product_metrics['price_elasticity'] = (
        product_metrics['quantity'].pct_change() / product_metrics['unit_price'].pct_change()
    )
    product_metrics['price_elasticity'].replace([np.inf, -np.inf], np.nan, inplace=True)
    product_metrics['price_elasticity'].fillna(0, inplace=True)

    return product_metrics



def analyze_time(Transactions):
    logging.info("Starting time-based analysis")

    # Hourly analysis
    Transactions['hour'] = pd.to_datetime(Transactions['date_time']).dt.hour
    hourly_analysis = Transactions.groupby('hour').agg(
        transaction_count=('transaction_id', 'count'),
        avg_transaction_value=('final_total', 'mean')
    ).reset_index()

    # Seasonal analysis
    Transactions['season'] = pd.to_datetime(Transactions['date_time']).dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    seasonal_analysis = Transactions.groupby('season').agg(
        transaction_count=('transaction_id', 'count'),
        avg_transaction_value=('final_total', 'mean')
    ).reset_index()

    logging.info("Time-based analysis completed")
    return {
        'hourly_trends': hourly_analysis.to_dict('records'),
        'seasonal_trends': seasonal_analysis.to_dict('records')
    }

def perform_analysis(filepath):
    logging.info("Starting analysis on uploaded file")

    # Read Excel file
    excel_file = pd.ExcelFile(filepath)

    # Load required sheets
    Customers = excel_file.parse("Customers")
    Transactions = excel_file.parse("Transactions")
    Customer_Feedback = excel_file.parse("Customer_Feedback")
    Order_Details = excel_file.parse("Order_Details")

    # Print debug information
    logging.info(f"Customers columns: {Customers.columns.tolist()}")
    logging.info(f"Transactions columns: {Transactions.columns.tolist()}")
    logging.info(f"Customer_Feedback columns: {Customer_Feedback.columns.tolist()}")
    logging.info(f"Order_Details columns: {Order_Details.columns.tolist()}")

    # Perform all analyses
    customer_metrics = analyze_customers(Customers, Transactions)
    store_metrics = analyze_store_performance(Transactions, Customer_Feedback)
    product_metrics = analyze_products(Order_Details, Customer_Feedback, Transactions)
    time_metrics = analyze_time(Transactions)

    # Add new analyses
    rfm_analysis = perform_rfm_analysis(Customers, Transactions)
    product_affinity = analyze_product_affinity(Order_Details)
    store_efficiency = analyze_store_efficiency(Transactions, 
                                              store_sizes={"Store1": 1000, "Store2": 1200})  # Update with your actual store sizes
    price_elasticity = calculate_price_elasticity(Order_Details)

    # Combine all metrics
    results = {
        **customer_metrics,
        **store_metrics,
        **product_metrics,
        **time_metrics,
        'customer_segments': rfm_analysis['customer_segment'].value_counts().to_dict(),
        'top_product_pairs': product_affinity.to_dict('records'),
        'store_efficiency': store_efficiency.to_dict('records'),
        'price_elasticity': price_elasticity.to_dict('records')
    }

    logging.info("Analysis completed")
    return results


@app.route('/')
def home():
    return render_template("upload.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = 'uploaded_file.xlsx'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info("File uploaded successfully")
        return jsonify({"message": "File uploaded successfully", "status": "success"})

@app.route('/begin_analysis', methods=['GET'])
def begin_analysis():
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.xlsx')
        if not os.path.exists(filepath):
            return jsonify({"error": "No file found"}), 404

        results = perform_analysis(filepath)
        return jsonify({
            "message": "Analysis completed successfully",
            "results": results
        })

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
