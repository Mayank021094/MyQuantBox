# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3

# ---------------------CONSTANTS------------------#
db = sqlite3.connect("./raw_datasets/head_database.db")
cursor = db.cursor()
DB_user = sqlite3.connect("./raw_datasets/user_db.db")

# --------------------MAIN CODE-------------------#

# Creating Base Database For Stocks
stock_bucket = pd.read_csv(r'./raw_datasets/stock_buckets.csv')


variables = stock_bucket.columns
# Iterate over each column in the DataFrame
for i in variables:
    table_name = f"table_{i}"
    table_values = stock_bucket[i].dropna().tolist()  # Drop NaN values if any

    # Create the table if it does not exist
    if i == 'industry':
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({i} TEXT NOT NULL)")
    else:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({i} TEXT UNIQUE NOT NULL)")

    # Insert values into the table
    for value in table_values:
        cursor.execute(f"INSERT OR IGNORE INTO {table_name} ({i}) VALUES (?)", (value,))

# Commit the transaction and close the connection
db.commit()
db.close()

# Creating Database for Users
try:
    cursor = DB_user.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_db (
            id INTEGER PRIMARY KEY,
            fname TEXT NOT NULL,
            lname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            birth_date TEXT NOT NULL  -- Storing as TEXT in 'YYYY-MM-DD' format
        )
    """)

    DB_user.commit()

except Exception as e:
    print("An error occurred:", e)
    DB_user.rollback()  # Roll back any changes if an error occurs

finally:
    DB_user.close()  # Ensure the database connection is closed
