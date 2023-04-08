# importing relavent packages
import numpy as np
import pandas as pd
from os import listdir
## function for data loading
def DataETL(datapath):
    """
    DataETL loads experimental data from all experimental data files (xlsx).
    Filters data and keeps only important columns.
    Combine selected data and save it to csv file.

    useage: DataETL('datapath')
    """
    data_filenames_list = listdir(datapath)
    df_frames = []
    for i in range(0, len(data_filenames_list)) :
        df_raw = pd.read_excel(datapath + data_filenames_list[0])
        selected_columns = ['Time (Min)', 'Tc - AVG 1&2 (oC)', 'Te - AVG 1,2,3 (oC)', 'Pressure (mm of Hg).1', 'Te - Tc (oC)', 'Q (W)',
            'Resistance (oC/W)']
        df_selected_columns = df_raw[selected_columns]
        df_frames.append(df_selected_columns)
        df = pd.concat(df_frames, axis=0, ignore_index=True)
        df_out = df.to_csv(datapath + "combined_data.csv")
    return df
