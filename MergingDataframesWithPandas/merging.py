# DataCamp
# Merging Manipulating dataframe with Pandas


# Import pandas
import pandas as pd
# Read 'Bronze.csv' into a DataFrame: bronze
bronze = pd.read_csv('Bronze.csv')
# Read 'Silver.csv' into a DataFrame: silver
silver = pd.read_csv('Silver.csv')
# Read 'Gold.csv' into a DataFrame: gold
gold = pd.read_csv('Gold.csv')
# Print the first five rows of gold
print(gold.head())


# Import pandas
import pandas as pd
# Create the list of file names: filenames
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']
# Create the list of three DataFrames: dataframes
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))
# Print top 5 rows of 1st DataFrame in dataframes
print(dataframes[0].head())

# Make a copy of gold: medals
medals = gold.copy()
# Create list of new column labels: new_labels
new_labels = ['NOC', 'Country', 'Gold']
# Rename the columns of medals using new_labels
medals.columns = new_labels
# Add columns 'Silver' & 'Bronze' to medals
medals['Silver'] = silver['Total']
medals['Bronze'] = bronze['Total']
# Print the head of medals
print(medals.head())



# Import pandas
import pandas as pd
# Read 'monthly_max_temp.csv' into a DataFrame: weather1
weather1 = pd.read_csv('monthly_max_temp.csv', index_col='Month')
# Print the head of weather1
print(weather1.head())
# Sort the index of weather1 in alphabetical order: weather2
weather2 = weather1.sort_index()
# Print the head of weather2
print(weather2.head())
# Sort the index of weather1 in reverse alphabetical order: weather3
weather3 = weather1.sort_index(ascending=False)
# Print the head of weather3
print(weather3.head())
# Sort the index of weather1 numerically using the values of 'Max TemperatureF': weather4
weather4 = weather1.sort_values('Max TemperatureF')
# Print the head of weather4
print(weather4.head())


# Import pandas
import pandas as pd
# Reindex weather1 using the list year: weather2
weather2 = weather1.reindex(year)
# Print weather2
print(weather2)
# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1.reindex(year).ffill()
# Print weather3
print(weather3)


names_1981 = pd.read_csv('names1981.csv', header=None, names=['name','gender','count'], index_col=(0,1))
names_1881 = pd.read_csv('names1881.csv', header=None, names=['name','gender','count'], index_col=(0,1))
# Import pandas
import pandas as pd
# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)
# Print shape of common_names
print(common_names.shape)
# Drop rows with null counts: common_names
common_names = common_names.dropna()
# Print shape of new common_names
print(common_names.shape)


# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]
# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5 / 9
# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')
# Print first 5 rows of temps_c
print(temps_c.head())



import pandas as pd
# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP.csv', parse_dates=True, index_col='DATE')
# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':, :]
# Print the last 8 rows of post2008
print(post2008.tail(8))
# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()
# Print yearly
print(yearly)
# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100
# Print yearly again
print(yearly)


# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', parse_dates=True, index_col='Date')
# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]
# Print the head of dollars
print(dollars.head())
# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis='rows')
# Print the head of pounds
print(pounds.head())


# Import pandas
import pandas as pd
# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('sales-jan-2015.csv', parse_dates=True, index_col='Date')
# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('sales-feb-2015.csv', parse_dates=True, index_col='Date')
# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('sales-mar-2015.csv', parse_dates=True, index_col='Date')
# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']
# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']
# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']
# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)
# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])
# Compute & print total sales in quarter1
print(quarter1.sum())


# Initialize empty list: units
units = []

# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month['Units'])
# Concatenate the list: quarter1
quarter1 = pd.concat(units, axis='rows')
# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])


# Add 'year' column to names_1881 and names_1981
names_1881['year'] = 1881
names_1981['year'] = 1981
# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981, ignore_index=True)
# Print shapes of names_1981, names_1881, and combined_names
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)
# Print all rows that contain the name 'Morgan'
print(combined_names.loc[combined_names['name'] == 'Morgan'])


# Concatenate weather_max and weather_mean horizontally: weather
weather = pd.concat([weather_max, weather_mean], axis=1)
# Print weather
print(weather)


for medal in medal_types:
    # Create the file name: file_name
    file_name = "%s_top5.csv" % medal
    # Create list of column names: columns
    columns = ['Country', medal]
    # Read file_name into a DataFrame: df
    medal_df = pd.read_csv(file_name, header=0, index_col='Country', names=columns)
    # Append medal_df to medals
    medals.append(medal_df)
# Concatenate medals horizontally: medals
medals = pd.concat(medals, axis='columns')
# Print medals
print(medals)


for medal in medal_types:
    file_name = "%s_top5.csv" % medal
    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name, index_col='Country')
    # Append medal_df to medals
    medals.append(medal_df)
# Concatenate medals: medals
medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'])
# Print medals in entirety
print(medals)


# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)
# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])
# Print data about silver medals
print(medals_sorted.loc['silver'])
# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice
# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:, 'United Kingdom'], :])


# Concatenate dataframes: february
february = pd.concat(dataframes, keys=['Hardware', 'Software', 'Service'], axis=1)
# Print february.info()
print(february.info())
# Assign pd.IndexSlice: idx
idx = pd.IndexSlice
# Create the slice: slice_2_8
slice_2_8 = february.loc['feb-2-2015':'feb-8-2015', idx[:, 'Company']]
# Print slice_2_8
print(slice_2_8)

# Concatenation and joins

# Create the list of DataFrames: medal_list
medal_list = [bronze, silver, gold]
# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list, keys=['bronze', 'silver', 'gold'], axis=1, join='inner')
# Print medals
print(medals)


# Resample and tidy china: china_annual
china_annual = china.resample('A').pct_change(10).dropna()
# Resample and tidy us: us_annual
us_annual = us.resample('A').pct_change(10).dropna()
# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual, us_annual], axis=1, join='inner')
# Resample gdp and print
print(gdp.resample('10A').last())


# Merging

pd.merge(revenue, managers, on='city')

# Merge revenue with managers on 'city': merge_by_city
merge_by_city = pd.merge(revenue, managers, on='city')
# Print merge_by_city
print(merge_by_city)
# Merge revenue with managers on 'branch_id': merge_by_id
merge_by_id = pd.merge(revenue, managers, on='branch_id')
# Print merge_by_id
print(merge_by_id)


# Merge revenue & managers on 'city' & 'branch': combined
combined = pd.merge(revenue, managers, left_on='city', right_on='branch')
# Print combined
print(combined)


# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX','CO','IL','CA']
# Add 'state' column to managers: managers['state']
managers['state'] = ['TX','CO','CA','MO']
# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers, on=['branch_id', 'city', 'state'])
# Print combined
print(combined)


managers.join(revenue, lsuffix='_mgn', rsuffix='_rev', how='left')


#
pd.merge(students, midterm_results, how='left').


# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(revenue, sales, how='right', on=['city', 'state'])

# Print revenue_and_sales
print(revenue_and_sales)
# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales, managers, how='left', left_on=['city', 'state'], right_on=['branch', 'state'])
# Print sales_and_managers
print(sales_and_managers)


# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers, revenue_and_sales)
# Print merge_default
print(merge_default)
# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers, revenue_and_sales, how='outer')
# Print merge_outer
print(merge_outer)
# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers, revenue_and_sales, on=['city','state'], how='outer')
# Print merge_outer_on
print(merge_outer_on)


# Perform the first ordered merge: tx_weather
tx_weather = pd.merge_ordered(austin, houston)
# Print tx_weather
print(tx_weather)
# Perform the second ordered merge: tx_weather_suff
tx_weather_suff = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus', '_hus'])
# Print tx_weather_suff
print(tx_weather_suff)
# Perform the third ordered merge: tx_weather_ffill
tx_weather_ffill = pd.merge_ordered(austin, houston, on='date', suffixes=['_aus', '_hus'], fill_method='ffill')
# Print tx_weather_ffill
print(tx_weather_ffill)


# Merge automobile and oil: merged
merged = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')
# Print the tail of merged
print(merged.tail())
# Resample merged: yearly
yearly = merged.resample('A', on='Date')[['mpg', 'Price']].mean()
# Print yearly
print(yearly)
# print yearly.corr()
print(yearly.corr())


