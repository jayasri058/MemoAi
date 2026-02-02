import sqlite3
import pandas as pd

def inspect_db():
    try:
        conn = sqlite3.connect('memoai.db')
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {[t[0] for t in tables]}")
        
        for table in tables:
            table_name = table[0]
            print(f"\n--- Data in '{table_name}' (First 5 rows) ---")
            
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Get data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=columns)
                print(df.to_string(index=False))
            else:
                print("(Table is empty)")
                
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_db()
