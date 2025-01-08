import pandas as pd
from sqlalchemy import create_engine

def upload_data_to_postgresql(file_path):
    # PostgreSQL External Database URL
    db_url = "postgresql://verticals_psql_user:HiHvIw1VI6jY5iHCr0xutf57kQAZDcLt@dpg-ctvg5sogph6c73dsjth0-a.oregon-postgres.render.com/verticals_psql"

    # Create database engine
    engine = create_engine(db_url)

    # Load the Excel file with all sheets
    excel_data = pd.ExcelFile(file_path)

    # Define table mappings
    table_mappings = {
        "Menu": "menu_table",
        "Customers": "customers_table",
        "Transactions": "transactions_table",
        "Order_Details": "order_details_table",
        "Customer_Feedback": "feedback_table",
        "Store_Locations": "stores_table",
        "Summary_Statistics": "summary_statistics_table",
        "Customer_Segments": "customer_segments_table",
        "Store_Performance": "store_performance_table",
        "Product_Analysis": "product_analysis_table",
        "Time_Analysis": "time_analysis_table"
    }

    # Iterate over each sheet and upload to corresponding table
    for sheet_name, table_name in table_mappings.items():
        if sheet_name in excel_data.sheet_names:
            print(f"Uploading {sheet_name} to {table_name}...")
            df = excel_data.parse(sheet_name)

            # Save to the database, replacing if table exists
            try:
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                print(f"Successfully uploaded {sheet_name} to {table_name}.")
            except Exception as e:
                print(f"Failed to upload {sheet_name} to {table_name}: {e}")

if __name__ == "__main__":
    # Path to the generated Excel file
    file_path = "complete_cafe_analytics.xlsx"
    
    # Upload data to PostgreSQL
    upload_data_to_postgresql(file_path)
