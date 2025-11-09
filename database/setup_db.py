import sqlite3

# Create (or connect to) database
conn = sqlite3.connect('database/data.db')
c = conn.cursor()

# Create table
c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()
print("✅ Database and table created successfully.")
