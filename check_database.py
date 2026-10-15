import sqlite3

connection = sqlite3.connect("data/index.sqlite")
tables = connection.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
)

print("Database tables:")
for table in tables:
    print("-", table[0])

connection.close()
