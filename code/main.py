import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



df = pd.read_csv("../data/trending_searches_in_us.csv")


os.makedirs("../images", exist_ok=True)



print("FIRST 10 ROWS:")
print(df.head(10))

print("\nDATASET INFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe(include='all'))



search_column = "query"

top_searches = df[search_column].value_counts().head(10)

print("\nTOP TRENDING SEARCHES:")
print(top_searches)

plt.figure(figsize=(10,5))
top_searches.plot(kind='bar')

plt.title("Top Trending Searches in US")
plt.xlabel("Search Term")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("../images/top_trending_searches.png")
plt.show()



column = "search_volume"

data = df[column].dropna()

mu = data.mean()
sigma = data.std(ddof=0)

z_scores = (data - mu) / sigma

df["Z_Score"] = z_scores

print("\nZ-SCORE SAMPLE:")
print(df[[column, "Z_Score"]].head(10))



plt.figure(figsize=(8,5))

plt.hist(data, bins=30, density=True, alpha=0.6)

x = np.linspace(data.min(), data.max(), 100)
normal_curve = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

plt.plot(x, normal_curve)

plt.title("Search Volume Distribution (Raw Data)")
plt.xlabel("Search Volume")
plt.ylabel("Density")

plt.savefig("../images/search_volume_distribution.png")
plt.show()



plt.figure(figsize=(8,5))

plt.hist(z_scores, bins=30, density=True, alpha=0.7)

plt.title("Z-Score Distribution (Standardized Data)")
plt.xlabel("Z-Score")
plt.ylabel("Density")



print("\nDONE: All charts saved successfully.")