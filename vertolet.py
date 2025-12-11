import pandas as pd

df = pd.read_csv("vertolet.csv")

# Базовая очистка
df = df.dropna(how="all")
df = df.drop_duplicates()

# Типы данных
df["trans_date"] = pd.to_datetime(df["trans_date"], errors="coerce")
df["total_sum"] = pd.to_numeric(df["total_sum"], errors="coerce")

# Критичные поля
df = df.dropna(subset=["transaction_code", "trans_date", "total_sum", "bank_name"])

# Текстовые поля
df["bank_name"] = df["bank_name"].str.strip().str.upper()
df["emitent_region"] = df["emitent_region"].str.strip().str.title()
df["gender"] = df["gender"].str.strip().str.lower()

# Флаги
df["p2p_flag"] = df["p2p_flag"].fillna(0).astype(int)

# Фильтр выбросов
df = df[df["total_sum"] > 0]
low, high = df["total_sum"].quantile([0.01, 0.99])
df = df[(df["total_sum"] >= low) & (df["total_sum"] <= high)]

# Сохранение
df.to_csv("cleaned_vertolet.csv", index=False)
