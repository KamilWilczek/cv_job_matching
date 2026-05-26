"""Quick connectivity test: PostgreSQL + pgvector (psycopg v3)."""
import os
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
register_vector(conn)
cur = conn.cursor()

cur.execute("SELECT version();")
print("PostgreSQL:", cur.fetchone()[0][:60])

cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
row = cur.fetchone()
print(f"pgvector extension: {row[0]} v{row[1]}")

# Test vector operations directly
cur.execute("CREATE TEMP TABLE _test (id INT, v VECTOR(3));")
cur.execute("INSERT INTO _test VALUES (1, '[1,0,0]'), (2, '[0,1,0]'), (3, '[0,0,1]');")
cur.execute("SELECT id, v <=> '[1,0,0]' AS distance FROM _test ORDER BY distance;")
rows = cur.fetchall()
print("\nVector search test (distance from [1,0,0]):")
for id_, dist in rows:
    print(f"  id={id_}  distance={float(dist):.4f}")

print("\nAll checks passed.")
conn.close()
