import pandas as pd

# Flexible paths
data_dir = '../P2P_PPDAI_DATA/'
lc_path = f'{data_dir}LC.csv'
lp_path = f'{data_dir}LP.csv'
lcis_path = f'{data_dir}LCIS.csv'

# Load with try-except
try:
    lc = pd.read_csv(lc_path)
    print("LC loaded successfully")
except Exception as e:
    print(f"Error loading LC: {e}")

# Inspect LC
print("LC Head:\n", lc.head())
print("\nLC Info:")
lc.info()
print("\nLC Shape:", lc.shape)
print("\nLC Describe:\n", lc.describe())
print("\nLC Missing:\n", lc.isnull().sum())
print("\nLC Duplicates:", lc.duplicated().sum())

# Inspect LP
print("LP Head:\n", lp.head())
print("\nLP Info:")
lp.info()
print("\nLP Shape:", lp.shape)
print("\nLP Describe:\n", lp.describe())
print("\nLP Missing:\n", lp.isnull().sum())
print("\nLP Duplicates:", lp.duplicated().sum())

# Inspect LCIS
print("LCIS Head:\n", lcis.head())
print("\nLCIS Info:")
lcis.info()
print("\nLCIS Shape:", lcis.shape)
print("\nLCIS Describe:\n", lcis.describe())
print("\nLCIS Missing:\n", lcis.isnull().sum())
print("\nLCIS Duplicates:", lcis.duplicated().sum())
