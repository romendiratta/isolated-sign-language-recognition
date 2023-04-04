import pandas as pd
# Read Parquet file
landmark_df = pd.read_parquet("/home/dpakapd/Documents/Berkeley/W251 Deep Learning in Edge/Final Project/Edge Device/Safekeeping/landmarks.parquet")
#print(landmark_df.tail(40))
# Calculate the min and max of X and Y in order to scale the values
min_x = landmark_df.x.min()
max_x = landmark_df.x.max()
min_y = landmark_df.y.min()
max_y = landmark_df.y.max()
#print(min_y, max_y)

def scale_x(val):
    return (val - min_x) / (max_x - min_x)

def scale_y(val):
    return (val - min_y) / (max_y - min_y)

landmark_df = landmark_df.dropna()
# Pass the dataframe X and Y values to the scale function
landmark_df['scaled_x'] = landmark_df.x.apply(scale_x)
landmark_df['scaled_y'] = landmark_df.y.apply(scale_y)

landmark_df = landmark_df.drop(columns=['x','y'])

#Create new column called landmark that adds 'type' - 'landmark_index'
landmark_df['landmark'] = landmark_df['type'] +" - "+landmark_df['landmark_index'].astype(str)

landmark_df = landmark_df.drop(columns=['row-id','type','landmark_index'])
grouped_data = landmark_df.groupby(['frame','landmark'])[['scaled_x','scaled_y']].sum().reset_index()
#pivoted_data = grouped_data.pivot_table(index = 'frame',columns='landmark',values=['scaled_x','scaled_y'],fill_value=0).reset_index(inplace=True)
#pivoted_data.columns = ['-'.join(col).strip() for col in pivoted_data.columns.values]
print(grouped_data)