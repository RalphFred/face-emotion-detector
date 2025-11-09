import sqlite3

# Connect to your database
conn = sqlite3.connect('database/data.db')
c = conn.cursor()

# Delete all rows from the predictions table
c.execute("DELETE FROM predictions")

# (Optional) Reset the auto-increment counter
c.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")

# Save changes and close
conn.commit()
conn.close()

print("✅ All data has been cleared from the database.")
