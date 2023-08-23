import pandas as pd

# df structure: store_id,date,client_id,product_id,price

# Load dataframe and add weekday feature
df = pd.read_csv('../data/train-2023-08-01.csv')

# Converting date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extracting year, month, day and weekday
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# Group by store_id, year, month, day and weekday
df_result = df.groupby(['store_id', 'year', 'month', 'day', 'weekday'])['product_id'].count().reset_index()
df_result.rename(columns={'product_id': 'total_sales'}, inplace=True)

print(df_result.head())

# Save the result using folder ../data (relative to src) using parquet file format.
df_result.to_parquet('../data/train-2023-08-01.parquet', index=False)