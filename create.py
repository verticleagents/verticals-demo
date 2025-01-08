import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_complete_cafe_dataset(num_transactions=15000):
    # Keep existing menu_items and stores dictionaries
    menu_items = {
        'TWC_SPECIALS': {
            'Sea Salt Mocha': {'base_price': 310, 'sizes': {'R': 310, 'M': 340, 'L': 395}, 'type': 'hot_cold', 'category': 'beverage'},
            'Orange Zest Mocha': {'base_price': 310, 'sizes': {'R': 310, 'M': 345, 'L': 395}, 'type': 'hot_cold', 'category': 'beverage'},
            'French Vanilla Latte': {'base_price': 309, 'sizes': {'R': 309, 'M': 345, 'L': 395}, 'type': 'hot_cold', 'category': 'beverage'},
            'Dry Hazelnut Cappuccino': {'base_price': 315, 'sizes': {'R': 315, 'M': 349, 'L': 395}, 'type': 'hot_cold', 'category': 'beverage'},
            'Caramel Macchiato': {'base_price': 320, 'sizes': {'R': 320, 'M': 355, 'L': 405}, 'type': 'hot_cold', 'category': 'beverage'},
            'Toffee Nut Latte': {'base_price': 350, 'sizes': {'R': 350, 'M': 390, 'L': 450}, 'type': 'hot_cold', 'category': 'beverage'}
        },
        'CLASSICS': {
            'Americano': {'base_price': 229, 'sizes': {'R': 229, 'M': 255, 'L': 280}, 'type': 'hot_cold', 'category': 'beverage'},
            'Cappuccino': {'base_price': 239, 'sizes': {'R': 239, 'M': 265, 'L': 290}, 'type': 'hot_cold', 'category': 'beverage'},
            'Latte': {'base_price': 239, 'sizes': {'R': 239, 'M': 265, 'L': 290}, 'type': 'hot_cold', 'category': 'beverage'},
            'Hot Chocolate': {'base_price': 249, 'sizes': {'R': 249, 'M': 275, 'L': 300}, 'type': 'hot', 'category': 'beverage'}
        },
        'FOOD': {
            'Classic Cream Cheese Bagel': {'base_price': 265, 'sizes': {'R': 265}, 'type': 'veg', 'category': 'breakfast'},
            'English Cucumber & Cream Cheese Bagel': {'base_price': 265, 'sizes': {'R': 265}, 'type': 'veg', 'category': 'breakfast'},
            'Chilli Cheese Garlic Toast': {'base_price': 299, 'sizes': {'R': 299}, 'type': 'veg', 'category': 'snack'},
            'Hummus & Pita Platter': {'base_price': 305, 'sizes': {'R': 305}, 'type': 'veg', 'category': 'snack'},
            'Paneer Tikka Wrap': {'base_price': 335, 'sizes': {'R': 335}, 'type': 'veg', 'category': 'wrap'},
            'Tandoori Chicken Wrap': {'base_price': 335, 'sizes': {'R': 335}, 'type': 'non_veg', 'category': 'wrap'}
        }
    }

    # Expanded store locations
    stores = {
        'Mall_01': {'area': 'Urban', 'city': 'Mumbai', 'type': 'Mall', 'size': 'Large'},
        'HighStreet_02': {'area': 'Urban', 'city': 'Delhi', 'type': 'Street', 'size': 'Medium'},
        'Airport_03': {'area': 'Transit', 'city': 'Bangalore', 'type': 'Airport', 'size': 'Small'},
        'Corporate_04': {'area': 'Business', 'city': 'Pune', 'type': 'Office', 'size': 'Medium'},
        'Mall_05': {'area': 'Suburban', 'city': 'Chennai', 'type': 'Mall', 'size': 'Large'},
        'HighStreet_06': {'area': 'Tourist', 'city': 'Goa', 'type': 'Street', 'size': 'Small'},
        'Corporate_07': {'area': 'Business', 'city': 'Hyderabad', 'type': 'Office', 'size': 'Large'},
        'Mall_08': {'area': 'Urban', 'city': 'Kolkata', 'type': 'Mall', 'size': 'Medium'}
    }

    # Generate menu table (same as before)
    menu_data = []
    for category, items in menu_items.items():
        for item_name, details in items.items():
            for size, price in details['sizes'].items():
                menu_data.append({
                    'item_id': f"{category}_{item_name.replace(' ', '_')}_{size}",
                    'item_name': item_name,
                    'category': category,
                    'size': size,
                    'price': price,
                    'type': details['type'],
                    'item_category': details['category']
                })

    # Enhanced customer generation
    customers = []
    num_customers = int(num_transactions/4)  # Assuming average 4 transactions per customer
    
    # Create weighted choices for more realistic distribution
    age_weights = [0.15, 0.45, 0.30, 0.10]  # Weighted towards young adults
    loyalty_weights = [0.40, 0.30, 0.20, 0.10]  # Weighted towards no loyalty/bronze
    visit_weights = [0.10, 0.30, 0.40, 0.20]  # Weighted towards monthly/occasional visits
    
    for i in range(num_customers):
        # Use weighted choices for more realistic distributions
        age_group = np.random.choice(
            ['Teenager (13-19)', 'Young Adult (20-29)', 'Adult (30-49)', 'Senior (50+)'],
            p=age_weights
        )
        loyalty_tier = np.random.choice(
            ['None', 'Bronze', 'Silver', 'Gold'],
            p=loyalty_weights
        )
        visit_frequency = np.random.choice(
            ['Daily', 'Weekly', 'Monthly', 'Occasional'],
            p=visit_weights
        )
        
        customer = {
            'customer_id': f'CUST{i:05d}',
            'age_group': age_group,
            'gender': random.choice(['Male', 'Female', 'Non-Binary']),
            'customer_type': random.choice(['First-time', 'Regular', 'Tourist']),
            'join_date': datetime.now() - timedelta(days=random.randint(0, 730)),
            'loyalty_tier': loyalty_tier,
            'dietary_preferences': random.choice(['None', 'Vegan', 'Gluten-free', 'Nut-free', 'Dairy-free']),
            'accessibility_needs': random.choice(['None', 'Wheelchair Access', 'Braille Menu', 'Special Assistance']),
            'home_location': random.choice(['< 1km', '1-5km', '5-10km', '> 10km']),
            'preferred_store': random.choice(list(stores.keys())),
            'occupation': random.choice(['Student', 'Professional', 'Self-employed', 'Retired']),
            'visit_frequency': visit_frequency
        }
        customers.append(customer)

    # Enhanced transaction generation
    transactions = []
    orders = []
    feedback_data = []
    
    # Create time-based patterns
    peak_hours = [8, 12, 17]  # Morning, lunch, and evening peaks
    weekend_multiplier = 1.5  # More transactions on weekends
    
    for i in range(num_transactions):
        # Select customer with preference for regulars
        customer = random.choice(customers)
        
        # Generate transaction datetime with realistic patterns
        base_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        # Adjust hour based on peak times
        hour_weights = [0.5 + (3.0 if h in peak_hours else 0) for h in range(24)]
        hour_weights = [w/sum(hour_weights) for w in hour_weights]
        transaction_hour = np.random.choice(range(24), p=hour_weights)
        
        transaction_datetime = base_date.replace(
            hour=transaction_hour,
            minute=random.randint(0, 59)
        )
        
        # Adjust transaction probability based on day of week
        if transaction_datetime.weekday() >= 5:  # Weekend
            if random.random() > 1/weekend_multiplier:
                continue
        
        transaction = {
            'transaction_id': f'TRX{i:07d}',
            'customer_id': customer['customer_id'],
            'date_time': transaction_datetime,
            'store_location': customer['preferred_store'],
            'order_type': random.choice(['Dine-in', 'Takeaway', 'App Order', 'Website Order']),
            'group_size': np.random.choice([1, 2, 3, 4, 5, 6], p=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02]),
            'visit_purpose': random.choice(['Quick Coffee', 'Work/Study', 'Social', 'Meeting', 'Leisure']),
            'time_spent_mins': random.randint(5, 180),
            'wifi_used': random.choice([True, False]),
            'season': random.choice(['Spring', 'Summer', 'Fall', 'Winter']),
            'day_part': 'Morning Rush' if 6 <= transaction_hour < 10 else 
                       'Afternoon' if 10 <= transaction_hour < 16 else 
                       'Evening Chill' if 16 <= transaction_hour < 20 else 'Night'
        }

        # Generate orders with realistic patterns
        num_items = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        transaction_total = 0
        
        # Ensure at least one beverage per transaction
        beverage_items = [item for item in menu_data if item['item_category'] == 'beverage']
        food_items = [item for item in menu_data if item['item_category'] != 'beverage']
        
        # First item is always a beverage
        beverage = random.choice(beverage_items)
        orders.append({
            'transaction_id': transaction['transaction_id'],
            'item_id': beverage['item_id'],
            'item_name': beverage['item_name'],
            'category': beverage['category'],
            'size': beverage['size'],
            'quantity': 1,
            'unit_price': beverage['price'],
            'customization': random.choice(['None', 'Extra Shot', 'Sugar Free', 'Soy Milk', 'Almond Milk']),
            'is_promotional': random.choice([True, False])
        })
        
        # Add remaining items
        for _ in range(num_items - 1):
            menu_item = random.choice(menu_data)
            quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            
            order = {
                'transaction_id': transaction['transaction_id'],
                'item_id': menu_item['item_id'],
                'item_name': menu_item['item_name'],
                'category': menu_item['category'],
                'size': menu_item['size'],
                'quantity': quantity,
                'unit_price': menu_item['price'],
                'customization': random.choice(['None', 'Extra Shot', 'Sugar Free', 'Soy Milk', 'Almond Milk']),
                'is_promotional': random.choice([True, False])
            }
            
            order['total_price'] = order['unit_price'] * order['quantity']
            transaction_total += order['total_price']
            orders.append(order)

        # Add transaction totals
        transaction['subtotal'] = transaction_total
        transaction['discount'] = round(transaction_total * (0.15 if customer['loyalty_tier'] == 'Gold' else 
                                                    0.10 if customer['loyalty_tier'] == 'Silver' else
                                                    0.05 if customer['loyalty_tier'] == 'Bronze' else 0), 2)
        transaction['final_total'] = transaction['subtotal'] - transaction['discount']
        
        # Generate feedback with higher probability for extreme experiences
        rating_probability = 0.35  # Increased feedback probability
        if random.random() < rating_probability:
            # Bias towards more positive ratings
            base_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.15, 0.30, 0.40])
            
            feedback = {
                'transaction_id': transaction['transaction_id'],
                'customer_id': customer['customer_id'],
                'rating_overall': base_rating,
                'rating_food': max(1, min(5, base_rating + random.randint(-1, 1))),
                'rating_service': max(1, min(5, base_rating + random.randint(-1, 1))),
                'rating_ambience': max(1, min(5, base_rating + random.randint(-1, 1))),
                'complaint_category': 'None' if base_rating >= 4 else random.choice(['Waiting Time', 'Product Quality', 'Pricing', 'Service']),
                'suggestion': random.choice(['Extend hours', 'More vegan options', 'Faster service', 'Better WiFi', 'More seating']),
                'feedback_channel': random.choice(['App', 'Website', 'In-Store', 'Social Media'])
            }
            feedback_data.append(feedback)
            
        transactions.append(transaction)

    # Create DataFrames
    menu_df = pd.DataFrame(menu_data)
    customers_df = pd.DataFrame(customers)
    transactions_df = pd.DataFrame(transactions)
    orders_df = pd.DataFrame(orders)
    feedback_df = pd.DataFrame(feedback_data)
    stores_df = pd.DataFrame([{'store_id': k, **v} for k, v in stores.items()])

    # Add total_spend to customers
    customer_totals = transactions_df.groupby('customer_id')['final_total'].sum().reset_index()
    customers_df = customers_df.merge(customer_totals, on='customer_id', how='left')
    customers_df.rename(columns={'final_total': 'total_spend'}, inplace=True)

    # Export to Excel

    # Export to Excel with enhanced summary statistics
    with pd.ExcelWriter('complete_cafe_analytics.xlsx', engine='openpyxl') as writer:
        # Main data sheets
        menu_df.to_excel(writer, sheet_name='Menu', index=False)
        customers_df.to_excel(writer, sheet_name='Customers', index=False)
        transactions_df.to_excel(writer, sheet_name='Transactions', index=False)
        orders_df.to_excel(writer, sheet_name='Order_Details', index=False)
        feedback_df.to_excel(writer, sheet_name='Customer_Feedback', index=False)
        stores_df.to_excel(writer, sheet_name='Store_Locations', index=False)

        # Enhanced Summary Statistics
        summary_stats = pd.DataFrame({
            'Metric': [
                # Customer Metrics
                'Total Customers',
                'Average Customer Spend',
                'Loyalty Program Participation Rate',
                'Top Customer Segment',
                'Most Common Visit Frequency',
                
                # Transaction Metrics
                'Total Transactions',
                'Average Transaction Value',
                'Peak Transaction Hour',
                'Most Popular Day Part',
                'Average Group Size',
                
                # Product Metrics
                'Most Popular Item',
                'Most Popular Category',
                'Average Items Per Transaction',
                'Most Common Customization',
                
                # Store Metrics
                'Most Popular Store',
                'Busiest Store Type',
                'Average Store Transaction Count',
                
                # Feedback Metrics
                'Feedback Response Rate',
                'Average Overall Rating',
                'Most Common Complaint',
                'Most Common Suggestion'
            ],
            'Value': [
                # Customer Metrics
                len(customers_df),
                f"₹{customers_df['total_spend'].mean():.2f}",
                f"{(customers_df['loyalty_tier'] != 'None').mean() * 100:.1f}%",
                customers_df['age_group'].mode()[0],
                customers_df['visit_frequency'].mode()[0],
                
                # Transaction Metrics
                len(transactions_df),
                f"₹{transactions_df['final_total'].mean():.2f}",
                transactions_df['date_time'].dt.hour.mode()[0],
                transactions_df['day_part'].mode()[0],
                f"{transactions_df['group_size'].mean():.1f}",
                
                # Product Metrics
                orders_df['item_name'].mode()[0],
                orders_df['category'].mode()[0],
                f"{len(orders_df) / len(transactions_df):.1f}",
                orders_df['customization'].mode()[0],
                
                # Store Metrics
                transactions_df['store_location'].mode()[0],
                stores_df['type'].mode()[0],
                f"{len(transactions_df) / len(stores_df):.0f}",
                
                # Feedback Metrics
                f"{(len(feedback_df) / len(transactions_df) * 100):.1f}%",
                f"{feedback_df['rating_overall'].mean():.1f}",
                feedback_df['complaint_category'].mode()[0],
                feedback_df['suggestion'].mode()[0]
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Additional Analysis Sheets
        
        # Customer Segments Analysis
        customer_segments = pd.DataFrame({
            'Segment': customers_df['age_group'].unique(),
            'Count': customers_df['age_group'].value_counts().values,
            'Avg_Spend': customers_df.groupby('age_group')['total_spend'].mean().values,
            'Most_Common_Visit_Time': customers_df.groupby('age_group')['visit_frequency'].agg(lambda x: x.mode()[0]).values
        })
        customer_segments.to_excel(writer, sheet_name='Customer_Segments', index=False)
        
        unique_stores = transactions_df['store_location'].unique()
        store_metrics = {
            'Store': [],
            'Total_Transactions': [],
            'Avg_Transaction_Value': [],
            'Most_Popular_Item': [],
            'Avg_Rating': []
        }
        
        for store in unique_stores:
            # Store name
            store_metrics['Store'].append(store)
            
            # Total transactions
            store_transactions = transactions_df[transactions_df['store_location'] == store]
            store_metrics['Total_Transactions'].append(len(store_transactions))
            
            # Average transaction value
            avg_value = store_transactions['final_total'].mean() if not store_transactions.empty else 0
            store_metrics['Avg_Transaction_Value'].append(avg_value)
            
            # Most popular item
            store_orders = orders_df[orders_df['transaction_id'].isin(store_transactions['transaction_id'])]
            most_popular = store_orders['item_name'].mode().iloc[0] if not store_orders.empty else 'N/A'
            store_metrics['Most_Popular_Item'].append(most_popular)
            
            # Average rating
            store_feedback = feedback_df[feedback_df['transaction_id'].isin(store_transactions['transaction_id'])]
            avg_rating = store_feedback['rating_overall'].mean() if not store_feedback.empty else 0
            store_metrics['Avg_Rating'].append(avg_rating)
        
        store_performance = pd.DataFrame(store_metrics)
        store_performance.to_excel(writer, sheet_name='Store_Performance', index=False)
        
        # Product Analysis - Fixed version with proper initialization
        unique_products = orders_df['item_name'].unique()
        product_metrics = {
            'Product': [],
            'Total_Orders': [],
            'Revenue': [],
            'Avg_Rating': []
        }
        
        for product in unique_products:
            # Product name
            product_metrics['Product'].append(product)
            
            # Total orders
            product_orders = orders_df[orders_df['item_name'] == product]
            product_metrics['Total_Orders'].append(product_orders['quantity'].sum())
            
            # Revenue
            revenue = (product_orders['unit_price'] * product_orders['quantity']).sum()
            product_metrics['Revenue'].append(revenue)
            
            # Average rating
            product_transactions = product_orders['transaction_id'].unique()
            product_feedback = feedback_df[feedback_df['transaction_id'].isin(product_transactions)]
            avg_rating = product_feedback['rating_overall'].mean() if not product_feedback.empty else 0
            product_metrics['Avg_Rating'].append(avg_rating)
        
        product_analysis = pd.DataFrame(product_metrics)
        product_analysis.to_excel(writer, sheet_name='Product_Analysis', index=False)
        
        # Time Analysis - Fixed version with proper initialization
        hours = list(range(24))
        time_metrics = {
            'Hour': [],
            'Transaction_Count': [],
            'Avg_Transaction_Value': [],
            'Most_Popular_Item': []
        }
        
        for hour in hours:
            # Hour
            time_metrics['Hour'].append(hour)
            
            # Transactions in this hour
            hour_transactions = transactions_df[transactions_df['date_time'].dt.hour == hour]
            time_metrics['Transaction_Count'].append(len(hour_transactions))
            
            # Average transaction value
            avg_value = hour_transactions['final_total'].mean() if not hour_transactions.empty else 0
            time_metrics['Avg_Transaction_Value'].append(avg_value)
            
            # Most popular item
            hour_orders = orders_df[orders_df['transaction_id'].isin(hour_transactions['transaction_id'])]
            most_popular = hour_orders['item_name'].mode().iloc[0] if not hour_orders.empty else 'N/A'
            time_metrics['Most_Popular_Item'].append(most_popular)
        
        time_analysis = pd.DataFrame(time_metrics)
        time_analysis.to_excel(writer, sheet_name='Time_Analysis', index=False)

    return menu_df, customers_df, transactions_df, orders_df, feedback_df, stores_df

# Generate the dataset
menu_df, customers_df, transactions_df, orders_df, feedback_df, stores_df = create_complete_cafe_dataset()

# Print basic statistics to verify the data generation
print(f"Generated {len(transactions_df)} transactions")
print(f"Generated {len(customers_df)} customers")
print(f"Generated {len(orders_df)} order items")
print(f"Generated {len(feedback_df)} feedback entries")
