import sqlite3
db = sqlite3.connect('data/stocks.db')

# Kolommen van calculated_scores
cols = db.execute("PRAGMA table_info(calculated_scores)").fetchall()
print("Kolommen calculated_scores:", [c[1] for c in cols])

# Alle tickers
rows = db.execute("SELECT ticker FROM calculated_scores ORDER BY ticker").fetchall()
tickers = [r[0] for r in rows]
print(f"\nAantal tickers: {len(tickers)}")
print("Tickers:", tickers)
