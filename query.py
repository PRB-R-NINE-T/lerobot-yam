import duckdb

# Open a parquet file and run SQL
con = duckdb.connect()
df = con.execute("SELECT * FROM 'episode_000068.parquet' WHERE action[8:14] = ARRAY[0.0,0.0,0.0,0.0,0.0,0.0,0.0];").fetchdf()
print(df)
