## PHP Data Analysis and Plotting Class
import numpy as np
import pandas as pd
from os import listdir
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

## Data Analysis
class PulseHeatPipe:
    """
    ## PulseHeatPipe - Manually defined functions for Advanced Data Analysis and Machine Learning.

    ## Useage: 
    ### imorting the module
    from analysis import PulsHeatPipe
    ### creating the reference variable 
    analysis = PulseHaatPipe("datapath")
    ### for a class help 
    help(analysis)
    ### for a function help
    help(analysis.data_etl)
    ### using a function from the class
    df, df_conv = analysis.data_etl
    
    ## list of avilable functions
    1. data_etl
    2. gibbs_fe
    3. data_chop
    4. data_stat
    5. data_property_avg
    6. best_TP
    7. plot_all_data
    8. plot_Te_Tc
    9. plot_eu
    """
    def __init__(self, datapath:str):
        self.T_k = 273.15 # To convert in kelvin
        self.P_const = 750.062 # To convert in bar
        self.R_const = 8.314 # Real Gas constant
        self.dG_standard = 30.9 # dG of water
        self.P_standard = 1 # atomospher pressure
        self.datapath = datapath
        print(f"Data loaded from directory: {self.datapath}")

    # data ETL    
    def data_etl(self):
        """
        data_etl loads experimental data from all experimental data files (xlsx).
        Filters data and keeps only important columns.
        Combine selected data and save to csv file.
        Conver units to MKS [K, bar] system and save to csv file. 

        useage: analysis = PulseHeatPipe("path")
                df, df_conv = analysis.data_etl()
        """
        data_filenames_list = glob.glob((self.datapath + 'php_*.xlsx'))
        df_frames = []
        for i in range(0, len(data_filenames_list)) :
            # loading data in loop
            df_raw = pd.read_excel((data_filenames_list[i]))
            selected_columns = ['Time (Min)', 'Tc - AVG (oC)', 'Te - AVG (oC)', 'Pressure (mm of Hg)', 'Te - Tc (oC)', 'Q (W)','Resistance (oC/W)']
            df_selected_columns = df_raw[selected_columns]
            df_frames.append(df_selected_columns)
            df = pd.concat(df_frames, axis=0, ignore_index=True).dropna()
        # converting data to MKS
        df_conv_fram = [df['Time (Min)'], df['Te - AVG (oC)']+self.T_k, df['Tc - AVG (oC)']+self.T_k, df['Te - Tc (oC)'] , df['Pressure (mm of Hg)']/self.P_const, df['Resistance (oC/W)']]
        df_conv = pd.concat(df_conv_fram, axis=1, ignore_index=True).dropna()
        df_conv_columns = ['t(min)' ,'Te[K]', 'Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]']
        df_conv.columns = df_conv_columns
        # saving data to csv
        df_out = df.to_csv(self.datapath + "combined_data.csv")
        df_conv_out = df_conv.to_csv(self.datapath + "combined_converted_data.csv")
        print(f"Compiled and converted data is saved at: {self.datapath}'combined_converted_data.csv'")
        return df, df_conv
    
    # to calculate gibbs free energy at given (T[K],P[bar])
    def gibbs_fe(self, data:pd.DataFrame):
        """
        gibbs_fe calculates the chagne in the gibbs free energy at a given vacuum pressure and temperature.
        dG = dG' + RTln(P/P')
        here, R = 8.314 [J/molK]
        P and P' = Pressure [bar]
        T = Temperature [K]

        useage: df_gfe = analysis.gibbs_fe(data)
        """
        Te = (data['Te[K]']) 
        Tc = (data['Tc[K]'])  
        P_vacuum = (data['P[bar]']) # converting to bar
        dG_vacuume_Te = self.R_const * Te * np.log(P_vacuum/self.P_standard)
        dG_vacuume_Tc = self.R_const * Tc * np.log(P_vacuum/self.P_standard)
        dG = dG_vacuume_Te - dG_vacuume_Tc
        selected_columns = ['t(min)' ,'Te[K]', 'Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE[KJ/mol]', 'GFE_Tc[KJ/mol]', 'dG[KJ/mol]']
        data = pd.concat([data, dG_vacuume_Te, dG_vacuume_Tc, dG], axis=1, ignore_index=True)
        data.columns = selected_columns
        data_out = data.to_csv(self.datapath + "gfe_combined.csv")
        msg = print(f"Gibbs Free Energy calculated data saved at: {self.datapath}'gfe_combined.csv")
        return data
    
    # To select data from specific Te range
    def data_chop(self, data:pd.DataFrame, Tmin=300, Tmax=400):
        """ 
        data_chop function is used to chop the data for the selected temperature value from the Te[K] column.

        useage: data = analysis.data_chop(df, Tmin, Tmax)
        here, Tmin/Tmax is a suitable value (int) from the data.
        default values: Tmin=300, Tmax=400
        """
        Tmina = data['Te[K]'].min()
        Tmaxa = data['Te[K]'].max()
        assert Tmin < Tmax, f"Entered wrong values: Correct range [Tmin:{round(Tmina,4)}, Tmax:{round(Tmaxa,4)} ]"
        print(f"Optimal range of temperature(Te) for data selection: [Tmin:{round(Tmina,4)}, Tmax:{round(Tmaxa)}]")
        data_T = data[data['Te[K]'] <= Tmax]
        data_T = data_T[data_T['Te[K]'] >= Tmin]
        return data_T
    
        # data mixing and re-arranging
    def data_stat(self, data:pd.DataFrame):
        """
        data_stat sorts and arrange value by a group from the experimental data loaded with data_etl function, calculates mean and standard deviation of the grouped data.
        Calculated result will be stored at the location of data files.

        df_mean, df_std = analysis.data_stat(data)
        """
        df_mean = data.sort_values(by=['Te[K]']).groupby(['Te[K]'], as_index=False).mean()
        df_mean_out = df_mean.to_csv(self.datapath + 'combined_mean.csv')
        df_std = data.sort_values(by=['Te[K]']).groupby(['Te[K]'], as_index=False).std().dropna()
        df_std_out = df_std.to_csv(self.datapath + 'combined_std.csv')
        print(f"Calculated mean and standard deviation values saved at {self.datapath}'combined_mean.csv' and 'combined_std.csv'")
        return df_mean, df_std
    
    # prepare average values for all thermal properties
    def data_property_avg(self, df_mean:pd.DataFrame, df_std:pd.DataFrame):
        """
        data_property_avg calculates average values of measured thermal properties for the given experiment data.

        useage: analysis.data_property_avg(df_mean, df_std)
        """
        # avg values 
        Tc_avg = df_mean['Tc[K]'].mean()
        P_avg = df_mean['P[bar]'].mean()
        dT_avg = df_mean['dT[K]'].mean()
        TR_avg = df_mean['TR[K/W]'].mean()
        GFE_avg = df_mean['GFE[KJ/mol]'].mean()
        # std values
        Tc_std = df_std['Tc[K]'].mean()
        P_std = df_std['P[bar]'].mean()
        dT_std = df_std['dT[K]'].mean()
        TR_std = df_std['TR[K/W]'].mean()
        GFE_std = df_std['GFE[KJ/mol]'].mean()
        # calculated results
        msg = (f"Tc  average:     {round(Tc_avg,4)} +- {round(Tc_std,4)} [K]\n"
        f"P   average:     {round(P_avg,4)} +- {round(P_std,4)} [bar]\n"
        f"dT  average:     {round(dT_avg,4)} +- {round(dT_std,4)} [K]\n"
        f"TR  average:     {round(TR_avg,4)} +- {round(TR_std,4)} [K/W]\n"
        f"GFE average:     {round(GFE_avg,4)} +- {round(GFE_std,4)} [KJ/mol]\n");
        return print(msg)
    
    # find optimal G(T,P) of PHP
    def best_TP(self, data:pd.DataFrame):
        """ 
        best_TP finds best G(T,P) with lowest dG (Change in Gibbs Free Energy for Te->Tc values at constant Pressure) from the experimental dataset.

        useage: analysis.best_TP(data)
        """
        df_opt = data[data['dG[KJ/mol]'] == data['dG[KJ/mol]'].min()]
        df_opt_idx = df_opt.index
        Te_opt = data['Te[K]'].loc[df_opt_idx]
        dT_opt = data['dT[K]'].loc[df_opt_idx]
        P_opt = data['P[bar]'].loc[df_opt_idx]
        dG_opt = data['dG[KJ/mol]'].loc[df_opt_idx]
        GFE_opt = data['GFE[KJ/mol]'].loc[df_opt_idx]
        TR_opt = data['TR[K/W]'].loc[df_opt_idx]
        msg = (f'Optimal G(T,P) condition at lowest (optimal) dG[{round(dG_opt.iloc[0],4)}]\n'
               f'Te optimal:        {round(Te_opt.iloc[0],4)}[K] \n'
               f'P  optimal:        {round(P_opt.iloc[0],4)}[bar] \n'
               f'dT optimal:        {round(dT_opt.iloc[0],4)}[K] \n'
               f'TR optimal:        {round(TR_opt.iloc[0],4)}[K/W] \n'
               f'GFE optimal:       dG({round(Te_opt.iloc[0],4)}, {round(P_opt.iloc[0],4)}) = {round(GFE_opt.iloc[0],4)} [KJ/mol]\n');
        return print(msg)
    
## Data Visualisation
class DataVisualisation(PulseHeatPipe):
    """ ## Data Visualisation class to plot PHP data.

        ## useage: 
        ### importing module
        from analysis import DataVisualisation
        ### creating the reference variable
        visual = DataVisualisation('sample')
        ### data visualisation; eg. plotting all data
        visual.plot_all_data()
    """
    def __init__(self, sample: str):
        super().__init__(sample)
        self.sample = sample

    def plot_all_data(self, data:pd.DataFrame):
        """ Data Visualisation
            
            useage: visual.plot_all_data(data)
        """
        plt.figure(figsize=(10,5))
        sns.lineplot(data)
        plt.xlabel('Data')
        plt.ylabel('Properties')
        plt.title(f"All Data - {self.sample}")
        plt.legend()

    def plot_Te_Tc(self, data:pd.DataFrame):
        """ Data Visualisation
            
            useage: visual.plot_Te_Tc(data)
        """
        plt.figure(figsize=(10,5))
        plt.plot(data['Te[K]'], label = 'Te[K]')
        plt.plot(data['Tc[K]'], label = 'Tc[K]')
        plt.xlabel('Te[K]')
        plt.ylabel('Tc[K]')
        plt.title(f"Te[K] vs Tc[K] - {self.sample}")
        plt.legend()

    def plot_eu(self, df_mean:pd.DataFrame, df_std:pd.DataFrame, property:str, point='.k', eu='r'):
        """ Data Visualisation
            
            useage: visual.plot_eu(df_mean, df_std, property='Tc[K]', point='.k', eu='r')
                    here, choose value from property list: ['Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE[KJ/mol]', 'GFE_Tc[KJ/mol]', 'dG[KJ/mol]']
        """
        self.property = property
        self.xproperty = 'Te[K]'
        self.point = point
        self.eu = eu
        properties = ['Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE[KJ/mol]', 'GFE_Tc[KJ/mol]', 'dG[KJ/mol]']
        if self.property in properties:    
            plt.figure(figsize=(10,5));
            plt.plot(df_mean[self.xproperty], df_mean[self.property], self.point, label=self.property)
            idx = df_std.index
            df_mean_idx = df_mean.loc[idx]
            plt.fill_between(df_std[self.xproperty], df_mean_idx[self.property] - 2* df_std[self.property], df_mean_idx[self.property] + 2* df_std[self.property],color=self.eu, alpha=0.2, label='Expanded Uncertainty')
            plt.xlabel(self.xproperty)
            plt.ylabel(self.property)
            plt.title(f"Expanded Uncertainty - {self.sample}")
            plt.legend()
        else:
            print(f"Entered invalid value [{self.property}] of thermal property!\n")
            print(f"Select any correct value from: {properties}")
        return
    
    