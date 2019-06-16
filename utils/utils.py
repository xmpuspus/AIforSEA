import glob
import pandas as pd
import numpy as np

def load_from_directory(path):
    """
    
    Loads all data into dataframes and merges them into one.
    
    """
    all_files = glob.glob(path + "/*.csv")
    list_dir = []
    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        list_dir.append(df)
        
    concat_df = pd.concat(list_dir, axis=0, ignore_index=True)
    return concat_df
    
def process_data(data, n_samples=120):
    """
    Convert input data into single array of the first n seconds of each of the features/signals.
    """

    n = data.bookingID.drop_duplicates().shape[0]
    data_sampled_per_booking = (data.groupby('bookingID')
                                .apply(lambda x: x.sample(n_samples))
                                .drop('bookingID', axis=1)
                                .reset_index()
                                .sort_values(['bookingID', 'second']))
    
    feature_cols = ['Accuracy', 'Bearing',
       'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x',
       'gyro_y', 'gyro_z','Speed', 'acceleration', 'gyro']

    n_feats = len(feature_cols)
    
    arrs = (data_sampled_per_booking[['bookingID'] + feature_cols]
            .groupby('bookingID')
            .apply(lambda x: x.values.reshape(n_samples, n_feats))
            .values)
    
    reshaped_data = np.hstack(arrs).reshape(n, n_samples * n_feats)
    return reshaped_data