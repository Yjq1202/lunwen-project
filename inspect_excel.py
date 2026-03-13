import pandas as pd

file_path = "reaults_paper/TSGym-vs-SOTA-full_new.xlsx"
try:
    xl = pd.ExcelFile(file_path)
    df = xl.parse(xl.sheet_names[0], header=None) # Read without header to access raw rows
    
    # Row 0 has model names
    raw_models = df.iloc[0].dropna().tolist()
    print("Models in Excel:", raw_models)
    
    # Column 0 has dataset names
    # They are sparse (merged cells in Excel imply NaNs in pandas)
    raw_datasets = df.iloc[2:, 0].dropna().unique().tolist()
    print("Datasets in Excel:", raw_datasets)
    
except Exception as e:
    print(f"Error reading excel file: {e}")

