# DataCamp
# Python Data Science Toolbox 1


# Define count_entries()
def count_entries1(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    #Initialize an empty dictionary: cols_count
    cols_count = {}

    # Iterate over column names in args
    for col_name in args:

        # Extract column from DataFrame: col
        col = df[col_name]

        # Iterate over the column in dataframe
        for entry in col:

            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1

            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result11 = count_entries1(tweets_df, 'lang')

# Call count_entries(): result2
result12 = count_entries1(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)


# Define count_entries()
def count_entries2(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The dataframe does not have a ' + col_name + ' column.')

    # Initialize an empty dictionary: langs_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]

    # Iterate over the column in dataframe
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

        # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result21 = count_entries2(tweets_df, 'lang')

# Print result1
print(result1)
