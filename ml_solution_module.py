# importing pkgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import missingno as msno
import glob
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class MachineLearning:
    def __init__(self, path:str):
        """
        ## MachineLearning is a class to perform various operations related to Machine Learning practice.

        ## useage:
        ### from ml_solutions import MachineLearning
        ml = MachineLearning("path")
        ### for help
        help(ml)

        """
        self.path = path
        self.dir_result = 'ml_result'
        self.output_path = os.path.join(self.path + self.dir_result)
        isExist = os.path.exists(self.output_path)
        if not isExist:
            os.mkdir(self.output_path)
            print(f'{self.output_path} directory created.')
        else:
            print(f'{self.output_path} already exists and ML reuslts will be stored here.')

    def data_prep(self, csv_file:str, sample:str, fr:float):
        """
        data_prep is a method to add information about the type of the working fluid (nanofluid or water as a simple working fluid) and its filling ratio in the PHP setup.

        useage:
        df_prep = ml.data_prep("data/path_individual_file", "DI", "40")
        """
        self.csv_file = csv_file
        self.sample = sample
        self.fr = fr
        data = pd.read_csv(self.csv_file)
        dict = {"Fluid": self.sample, "FR": self.fr}
        data_fr = data.assign(**dict)
        data_fr = data_fr.drop(data_fr.columns[0], axis=1)
        output_csv = (f"all_combined_data_{self.sample}_{self.fr}.csv")
        data_fr_out_path = os.path.join(self.output_path, output_csv)
        data_fr_out = data_fr.to_csv(data_fr_out_path)
        print(f'Compiled data stored at {data_fr_out_path}')
        return data_fr
    
    def data_compile(self):
        """
        data_compile is a method to combine all prepared data (from MachineLearning.data_prep method) and save them to a csv file.

        useage:
        df_compiled  = data_compile()
        """
        file_list = glob.glob(os.path.join(self.output_path,"all_combined_*.csv"))
        df_frames = []
        combined_data_file = 'super_combined_data.csv' 
        for i in range(0, len(file_list)):
            data = pd.read_csv(file_list[i])
            df_frames.append(data)
            df_combined = pd.concat(df_frames, axis=0, ignore_index=True)
        data_combined_out_path = os.path.join(self.output_path, combined_data_file)
        df_out = df_combined.to_csv(data_combined_out_path)
        print(f"All data compiled in a single csv file and saved at: {self.output_path} as {combined_data_file}")
        return df_combined

    def etl_visual(self, df:pd.DataFrame, y_value='dG[KJ/mol]', hue='Fluid', point=['b','o']):
        """
        etl_visual is a method to plot (scatter plot) a selected data as a function of Te[C]

        useage:
        ml.etl_visual()
        here, 
        hue = 'Fluid' or 'FR'
        point = ['b', 'o'] or ['g', 'r']    
        """
        now =datetime.now()
        self.current_time = now.strftime("%H:%M:%S")
        self.x_value = 'Te[K]'
        self.y_value = y_value
        self.point = point
        self.hue = hue
        properties = ['Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE[KJ/mol]', 'GFE_Tc[KJ/mol]', 'dG[KJ/mol]']
        if self.x_value and y_value in properties:
            plt.figure(figsize=(8,5))
            sns.scatterplot(x=df[self.x_value], y=df[self.y_value], hue=df[self.hue], palette=self.point)
        
        else:
            print(f"Entered invalid value [{self.y_value}] of thermal property!\n")
            print(f"Select any correct value from: {properties}")
    
    def data_filter_dG(self, data:pd.DataFrame, cutoff=0):
        """
        data_filter is a method to remove outliers and irrelevant data from dataset. All positive value of dG[KJ/mol] will be removed by default.

        useage:
        ml.data_filter(data)
        """
        self.data = data
        self.cutoff = cutoff
        data_filtered = self.data[data['dG[KJ/mol]'] <= self.cutoff]
        return data_filtered

    def data_filter_Te(self, data:pd.DataFrame, cutoff=400):
        """
        data_filter is a method to remove outliers and irrelevant data from dataset. Data can be filtered on the basis of Te[K] value.

        useage:
        ml.data_filter(data)
        """
        self.data = data
        self.cutoff = cutoff
        data_filtered = self.data[data['Te[K]'] <= self.cutoff]
        return data_filtered

    def data_split(self, data:pd.DataFrame, x=['Te[K]', 'P[bar]', 'Fluid', 'FR'], y=['Tc[K]', 'TR[K/W]', 'dG[KJ/mol]']):
        """
        data_xy_split is a method to split the data in features (x) and labels (y) as well as for train and test split.
        default values for x=['Te[K]', 'P[bar]', 'Fluid', 'FR']

        useage:
        x_data, y_data = data_xy_split(data, x=['Te[K]', 'P[bar]', 'Fluid', 'FR'], y=['Tc[K]', 'TR[K/W]', 'dG[KJ/mol])
        """
        self.data = data
        self.x = x
        self.y = y
        self.x_data = data[self.x]
        self.y_data = data[self.y]
        x_train, x_test, y_train, y_test = train_test_split(self.x_data, self.y_data, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test
    
    def mae_error(self, prediction, y_test, para=['Tc[K]', 'TR[K/W]', 'dG[KJ/mol]']):
        """
        mae_error is a method to estimate the Mean Absolute error in the prediction by the ML model.

        useage:
        mae_error(prediction, y_test, para=['Tc[K]', 'TR[K/W]', 'dG[KJ/mol]'])
        """
        self.prediction = prediction
        self.y_test = y_test
        self.para = para
        for i in para:
            err = mean_absolute_error(prediction[i], y_test[i])
            print(f'Mean Absolute Error in the preddiction [{i}]: {round(err,4)}')
        
    def avg_error(self, prediction, y_test, para=['Tc[K]', 'TR[K/W]', 'dG[KJ/mol]']):
        """
        avg_error is a method to estimate the average [%] error in the prediction by the ML model.

        useage:
        avg_error(prediction, y_test, para=['Tc[K]', 'TR[K/W]', 'dG[KJ/mol]'])
        """
        self.prediction = prediction
        self.y_test = y_test
        self.para = para
        for i in para:
            err = (prediction[i].astype(float)/y_test[i].astype(float))
            avg_err = abs(err.mean()*100)
            print(f'Average prediction accuracy over a test data [{i}]: {round(avg_err,4)} [%]')

    def goodness_of_fit(self, prediction, y_test, k:int):
        """
        r2_score is a method to estimate R2-score and R2-adjusted score for the ML prediction.

        useage:
        goodness_of_fit(prediction, y_test)
        here, k = number of explanatory variables
        """
        self.prediction = prediction
        self.y_test = y_test
        r2 = r2_score(y_test, prediction)
        n = y_test.shape[0]
        self.k = k
        r2_adj = 1 - (((1-r2)*(n-1)) / (n-k-1))
        result = (f'R2 score: {round(r2,4)}\nR2-adjusted score: {round(r2_adj,4)}')
        print(result)


