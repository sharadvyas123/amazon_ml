import sqlite3, os, sys

sys.stdout.reconfigure(encoding='utf-8')

db_path = os.path.join("database", "amazon_ml.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

for t in tables:
    tname = t[0]
    print(f"\n=== Table: {tname} ===")
    cursor.execute(f"PRAGMA table_info({tname})")
    cols = cursor.fetchall()
    for c in cols:
        print(f"  {c}")
    cursor.execute(f"SELECT COUNT(*) FROM {tname}")
    print(f"  Row count: {cursor.fetchone()[0]}")

# List indexes
cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'")
indexes = cursor.fetchall()
print(f"\n=== Indexes ({len(indexes)}) ===")
for idx in indexes:
    print(f"  {idx[0]} on {idx[1]}: {idx[2]}")

# Sample ground_truth_pairs
print("\n=== Sample ground_truth_pairs ===")
cursor.execute("SELECT * FROM ground_truth_pairs LIMIT 5")
cols = [d[0] for d in cursor.description]
print("Columns:", cols)
for row in cursor.fetchall():
    print(f"  {row}")

# Sample ground_truth
print("\n=== Sample ground_truth ===")
cursor.execute("SELECT * FROM ground_truth LIMIT 5")
cols = [d[0] for d in cursor.description]
print("Columns:", cols)
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()
