import pandas as pd

# %%
df = pd.read_excel(r"/Users/lindazhong/Documents/Quant之怒/Black-Litterman-Model/MyCode/Data/Price_Data.xlsx")
df
# %%
df.set_index("Date", inplace = True)
df.index = range(len(df))
df
