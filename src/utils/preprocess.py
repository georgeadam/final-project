def get_numeric_col_indices(df):
    categorical_cols = df.select_dtypes(exclude=["float"]).columns

    numeric_cols = df.columns.difference(categorical_cols)
    numeric_col_indices = []

    for numeric_col in numeric_cols:
        index = df.columns.get_loc(numeric_col)
        numeric_col_indices.append(index)

    return numeric_col_indices
