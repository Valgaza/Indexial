import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Retrieve the URI we saved earlier
DB_URL = os.getenv("SUPABASE_DB_URL")

def create_dynamic_table(table_name, schema_sql):
    try:
        # Connect to Supabase directly as a Postgres DB
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # 1. Execute the CREATE TABLE command
        # This SQL will eventually come from Groq/Llama 3.1
        print(f"Creating table: {table_name}...")
        cur.execute(schema_sql)
        
        # 2. Commit the transaction (Required for DDL)
        conn.commit()
        print("Success! Table created.")

        # Clean up
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

# --- Simulation of your Pipeline ---
# Simulated output from Llama 3.1
simulated_table_name = "doc_8f9a2_financials" 
simulated_schema = f"""
CREATE TABLE IF NOT EXISTS {simulated_table_name} (
    id SERIAL PRIMARY KEY,
    revenue_year INTEGER,
    total_revenue_usd DECIMAL(15, 2),
    net_profit_margin DECIMAL(5, 2)
);
"""

create_dynamic_table(simulated_table_name, simulated_schema)