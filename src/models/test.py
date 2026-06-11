
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/train.parquet')

print('=== OUTLIERS EN FEATURES (método IQR) ===')
cols_analizar = [c for c in df.columns if any(x in c for x in 
    ['temp', 'precip', 'humid', 'heat', 'cases_lag', 'incidencia_lag'])]

for col in cols_analizar[:10]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        print(f'{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)')
