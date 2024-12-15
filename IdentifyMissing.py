import pandas as pd

# Using absolute path
df = pd.read_csv('C:/PGDAI/Programming Language AI/CA70/Creditcard/creditcardT.csv')
input_file = 'C:/PGDAI/Programming Language AI/CA70/Creditcard/creditcardT.csv'
# Or for Excel file
# df = pd.read_excel('C:/Users/YourUsername/Documents/your_project/data.xlsx')

print(df.head())

# Identify missing values
missing_values = df.isnull()  # Boolean DataFrame: True where missing values exist
print("\nMissing values in the DataFrame:")
print(missing_values)
df = df.fillna(df.mean())
# Count missing values per column
missing_count_per_column = df.isnull().sum()
print("\nCount of missing values per column:")
print(df)
output_file = 'C:/PGDAI/Programming Language AI/CA70/Creditcard/imputed_data.csv'  # Replace with your desired output file path
df.to_csv(output_file, index=False)

df1 = pd.read_csv(input_file)
df2 = pd.read_csv(output_file)

# Check if the shapes of both DataFrames are the same
if df1.shape != df2.shape:
    print("The two CSV files have different shapes. They cannot be compared directly.")
    print(f"Shape of file1: {df1.shape}")
    print(f"Shape of file2: {df2.shape}")
else:
    print("The shapes of the files are the same. Proceeding with comparison...")

    # Identify where the DataFrames differ
    diff = df1 != df2

    # Loop through the DataFrame and find the rows and columns that are different
    rows, cols = diff.any(axis=1), diff.any(axis=0)  # Identify rows and columns that have differences

    # Extract the row indexes and column names where there are differences
    differing_rows = diff.index[rows].tolist()
    differing_cols = diff.columns[cols].tolist()

    if differing_rows and differing_cols:
        print("\nRows and Columns with differences:")
        for row in differing_rows:
            for col in differing_cols:
                if df1.at[row, col] != df2.at[row, col]:
                    print(f"Difference at Row: {row} | Column: '{col}'")
    else:
        print("No differences found between the two CSV files.")

    diff_file = 'C:/PGDAI/Programming Language AI/CA70/Creditcard/diff_data.csv'  # Replace with your desired output file path
    df.to_csv(diff_file, index=False)

