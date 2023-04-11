import numpy as np
import pandas as pd
from os import listdir
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class mdf:
    """
    MDF - Manually defined functions for Advanced Data Analysis and Machine Learning.

    HELP: help(mdf.function)

    Author: Nirmal Parmar, PhD 
    
    """
    T_k = 273.15 # To convert in kelvin
    P_const = 750.062 # To convert in bar
    R_const = 8.314 # Real Gas constant
    dG_standard = 30.9 # dG of water
    P_standard = 1 # atomospher pressure

    def __init__(self, datapath: str):
        """ 
        Menualy Defined Functions (MDF)
        For more details: help(function_name)

        1. DataETL
        2. GibbsFE
        3. DataChop
        4. DataArrange
        5. DataProvAvg
        6. BestTP
        """
        self.datapath = datapath
        print(f"Loading data from: {datapath}")

    # data ETL
    def DataETL(datapath: str):
        """
        DataETL loads experimental data from all experimental data files (xlsx).
        Filters data and keeps only important columns.
        Combine selected data and save to csv file.
        Conver units to MKS [K, bar] system and save to csv file. 

        useage: df, df_conv = DataETL('datapath')
        """
        data_filenames_list = glob.glob((datapath + 'php_*.xlsx'))
        df_frames = []
        for i in range(0, len(data_filenames_list)) :
            # loading data in loop
            df_raw = pd.read_excel((data_filenames_list[i]))
            selected_columns = ['Time (Min)', 'Tc - AVG (oC)', 'Te - AVG (oC)', 'Pressure (mm of Hg)', 'Te - Tc (oC)', 'Q (W)','Resistance (oC/W)']
            df_selected_columns = df_raw[selected_columns]
            df_frames.append(df_selected_columns)
            df = pd.concat(df_frames, axis=0, ignore_index=True).dropna()
        # converting data to MKS
        df_conv_fram = [df['Time (Min)'], df['Te - AVG (oC)']+mdf.T_k, df['Tc - AVG (oC)']+mdf.T_k, df['Te - Tc (oC)'] , df['Pressure (mm of Hg)']/mdf.P_const, df['Resistance (oC/W)']]
        df_conv = pd.concat(df_conv_fram, axis=1, ignore_index=True).dropna()
        df_conv_columns = ['t(min)' ,'Te[K]', 'Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]']
        df_conv.columns = df_conv_columns
        # saving data to csv
        df_out = df.to_csv(datapath + "combined_data.csv")
        df_conv_out = df_conv.to_csv(datapath + "combined_converted_data.csv")
        print(f"Compiled and converted data is saved at: {datapath}'combined_converted_data.csv'")
        return df, df_conv
    
    # calculation of Gibbs Free Energy
    def GibbsFE(data, datapath:str):
        """
        GibbsFE calculates chagne in gibbs free energy at a given vacuum pressure and temperature of PHP
        dG = dG' + RTln(P/P')
        here, R = 8.314 [J/molK]
        P and P' = Pressure [bar]
        T = Temperature [K]

        useage: df = GibbsFE(data)
        """
        Te = (data['Te[K]']) 
        Tc = (data['Tc[K]'])  
        P_vacuum = (data['P[bar]']) # converting to bar
        dG_vacuume_Te = mdf.R_const * Te * np.log(P_vacuum/mdf.P_standard)
        dG_vacuume_Tc = mdf.R_const * Tc * np.log(P_vacuum/mdf.P_standard)
        dG = dG_vacuume_Te - dG_vacuume_Tc
        selected_columns = ['t(min)' ,'Te[K]', 'Tc[K]', 'dT[K]', 'P[bar]', 'TR[K/W]', 'GFE [KJ/mol]', 'GFE_Tc [KJ/mol]', 'dG [KJ/mol]']
        data = pd.concat([data, dG_vacuume_Te, dG_vacuume_Tc, dG], axis=1, ignore_index=True)
        data.columns = selected_columns
        data_out = data.to_csv(datapath + "gfe_combined.csv")
        msg = print(f"Gibbs Free Energy calculated data saved at: {datapath}'gfe_combined.csv")
        return data
    
    # To select data from specific Te range
    def DataChop(data, Tmin=300, Tmax=400):
        """ 
        DataChop function used to chop the data for the selected temperature value from the Te[K] column.

        useage: data = DataChop(df, Tmin, Tmax)
        here, Tmin/Tmax is a suitable value (int) from the data.
        default values: Tmin=300, Tmax=400
        """
        Tmina = data['Te[K]'].min()
        Tmaxa = data['Te[K]'].max()
        assert Tmin < Tmax, f"Entered wrong values: Correct range [Tmin:{Tmina}, Tmax:{Tmaxa} ]"
        print(f"Optimal range of temperature(Te) for data selection: [Tmin:{Tmina}, Tmax:{Tmaxa}]")
        data_T = data[data['Te[K]'] <= Tmax]
        data_T = data_T[data_T['Te[K]'] >= Tmin]
        return data_T
    
    # data mixing and re-arranging
    def DataArrange(data, path:str):
        """
        DataArrange sorts and arrange value by group from the experimental data loaded with DataETL function, calculates mean and standard deviation of the grouped data.
        Calculated result will be stored at the location of data file.

        df_mean, df_std = DataArrange(data, path)
        """
        df_mean = data.sort_values(by=['Te[K]']).groupby(['Te[K]'], as_index=False).mean()
        df_mean_out = df_mean.to_csv(path + 'combined_mean.csv')
        df_std = data.sort_values(by=['Te[K]']).groupby(['Te[K]'], as_index=False).std().dropna()
        df_std_out = df_std.to_csv(path + 'combined_std.csv')
        print(f"Calculated mean and standard deviation values saved at {path}'combined_mean.csv' and 'combined_std.csv'")
        return df_mean, df_std
    
    # prepare average values for all thermal properties
    def DataPropAvg(df_mean, df_std):
        """
        DataPropAvg calculates average values of measured thermal properties for given experiment data.

        useage: DataPropAvg(df_mean, df_std)
        """
        # avg values 
        Tc_avg = df_mean['Tc[K]'].mean()
        P_avg = df_mean['P[bar]'].mean()
        dT_avg = df_mean['dT[K]'].mean()
        TR_avg = df_mean['TR[K/W]'].mean()
        GFE_avg = df_mean['GFE [KJ/mol]'].mean()
        # std values
        Tc_std = df_std['Tc[K]'].mean()
        P_std = df_std['P[bar]'].mean()
        dT_std = df_std['dT[K]'].mean()
        TR_std = df_std['TR[K/W]'].mean()
        GFE_std = df_std['GFE [KJ/mol]'].mean()
        # calculated results
        msg = (f"Tc  average:     {round(Tc_avg,4)} +- {round(Tc_std,4)} [K]\n"
        f"P   average:     {round(P_avg,4)} +- {round(P_std,4)} [bar]\n"
        f"dT  average:     {round(dT_avg,4)} +- {round(dT_std,4)} [K]\n"
        f"TR  average:     {round(TR_avg,4)} +- {round(TR_std,4)} [K/W]\n"
        f"GFE average:     {round(GFE_avg,4)} +- {round(GFE_std,4)} [KJ/mol]\n");
        return print(msg)
    
    # find optimal G(T,P) of PHP

    def BestTP(data):
        """ 
        BestTP finds best G(T,P) with lowest dG (Change in Gibbs Free Energy for Te->Tc values at constant Pressure) from the experimental dataset.

        useage: BestTP(data)
        """

        df_opt = data[data['dG [KJ/mol]'] == data['dG [KJ/mol]'].min()]
        df_opt_idx = df_opt.index
        Te_opt = data['Te[K]'].loc[df_opt_idx]
        P_opt = data['P[bar]'].loc[df_opt_idx]
        dG_opt = data['dG [KJ/mol]'].loc[df_opt_idx]
        GFE_opt = data['GFE [KJ/mol]'].loc[df_opt_idx]

        msg = (f'Optimal G(T,P) at lowest dG [{round(dG_opt.iloc[0],4)}]\n'
               f'GFE G(T,P): G({round(Te_opt.iloc[0],4)},{round(P_opt.iloc[0],4)}) = {round(GFE_opt.iloc[0],4)} [KJ/mol]\n'
               f'Temprature Te [K]: {round(Te_opt.iloc[0],4)} \n'
               f'Pressure-Vacuume [bar]: {round(P_opt.iloc[0],4)} \n');
        return print(msg)
    
    def PlotAllData(data):
        """ Data plotting 
            Plotfunction(dt_gfe)
        """
        data.plot(figsize = (10,5),
                  title = 'All Data')
        return
    
    def PlotTempData(data):
        """ Data plotting 
            Plotfunction(df_gfe)
        """
        data[['Te[K]', 'Tc[K]']].plot(style = '.',
                                      figsize = (10,5),
                                      title = 'Tc & Te',
                                      xlabel='Data',
                                      ylabel='Temperature[K]')
        return
    
    def PlotEUTemp(df_mean, df_std):
        """ Data plotting 
            Plotfunction(df_mean, df_std)
        """
        plt.figure(figsize=(10,5));
        plt.plot(df_mean['Te[K]'].index, df_mean['Te[K]'], '.b', label='Te-avg')
        plt.plot(df_mean['Tc[K]'].index, df_mean['Tc[K]'], '.k', label='Tc-avg' )
        idx = df_std.index
        df_mean_idx = df_mean.loc[idx]
        plt.fill_between(df_std['Tc[K]'].index, df_mean_idx['Tc[K]'] - 2* df_std['Tc[K]'], df_mean_idx['Tc[K]'] + 2* df_std['Tc[K]'],color='r', alpha=0.2, label='Expanded Uncertainty')
        plt.xlabel('Data')
        plt.ylabel('Temperature[K]')
        plt.legend()
        return
    
    def PlotEUPres(df_mean, df_std):
        """ Data plotting 
            Plotfunction(df_mean, df_std)
        """
        plt.figure(figsize=(10,5));
        plt.plot(df_mean['P[bar]'].index, df_mean['P[bar]'], '.k', label='Pressure [bar]' )
        idx = df_std.index
        df_mean_idx = df_mean.loc[idx]
        plt.fill_between(df_std['P[bar]'].index, df_mean_idx['P[bar]'] - 2* df_std['P[bar]'], df_mean_idx['P[bar]'] + 2* df_std['P[bar]'],color='g', alpha=0.2, label='Expanded Uncertainty')
        plt.xlabel('Data')
        plt.ylabel('Pressure[bar]')
        plt.legend()
        return
    
    def PlotEUTR(df_mean, df_std):
        """ Data plotting 
            Plotfunction(df_mean, df_std)
        """
        plt.figure(figsize=(10,5));
        plt.plot(df_mean['Te[K]'], df_mean['TR[K/W]'], '.k', label='Thermal Resistance [C/W]' )
        idx = df_std.index
        df_mean_idx = df_mean.loc[idx]
        plt.fill_between(df_std['Te[K]'], df_mean_idx['TR[K/W]'] - 2* df_std['TR[K/W]'], df_mean_idx['TR[K/W]'] + 2* df_std['TR[K/W]'],color='m', alpha=0.2, label='Expanded Uncertainty')
        plt.xlabel('Temperature [K]')
        plt.ylabel('Thermal Resistance [K/C]')
        plt.legend()
        return
    
    def PlotEUTP(df_mean, df_std):
        """ Data plotting 
            Plotfunction(df_mean, df_std)
        """
        plt.figure(figsize=(10,5));
        plt.plot(df_mean['Te[K]'], df_mean['P[bar]'],'.g', label='Temperature[Te] vs Pressure')
        idx = df_std.index
        df_mean_idx = df_mean.loc[idx]
        plt.fill_between(df_mean_idx['Te[K]'],df_mean_idx['P[bar]'] - 2* df_std['P[bar]'], df_mean_idx['P[bar]'] + 2* df_std['P[bar]'], color='r', alpha=0.2, label='Expanded Uncertainty')
        plt.xlabel('Temperature-Te[K]')
        plt.ylabel('Pressure[bar]')
        plt.legend()
        return

    def PlotEUGFE(df_mean, df_std):
        """ Data plotting 
            Plotfunction(df_mean, df_std)
        """
        plt.figure(figsize=(10,5));
        plt.plot(df_mean['Te[K]'], df_mean['GFE [KJ/mol]'], '.k', label='dG-Te [KJ/mol]')
        plt.plot(df_mean['Te[K]'], df_mean['GFE_Tc [KJ/mol]'], '.r', label='dG-Tc [KJ/mol]')
        idx = df_std.index
        df_mean_idx = df_mean.loc[idx]
        plt.fill_between(df_mean_idx['Te[K]'],df_mean_idx['GFE [KJ/mol]'] - 2* df_std['GFE [KJ/mol]'], df_mean_idx['GFE [KJ/mol]'] + 2* df_std['GFE [KJ/mol]'], color='g', alpha=0.3, label='Expanded Uncertainty')
        plt.fill_between(df_mean_idx['Te[K]'],df_mean_idx['GFE_Tc [KJ/mol]'] - 2* df_std['GFE_Tc [KJ/mol]'], df_mean_idx['GFE_Tc [KJ/mol]'] + 2* df_std['GFE_Tc [KJ/mol]'], color='r', alpha=0.2, label='Expanded Uncertainty')
        plt.xlabel('Temperature - Te [K]')
        plt.ylabel('Change in Gibbs Free Energy [KJ/mol]')
        plt.legend()
        return
    
    def PlotEUdG(df_mean, df_std):
        """ Data plotting 
            Plotfunction(df_mean, df_std)
        """
        plt.figure(figsize=(10,5));
        plt.plot(df_mean['Te[K]'], df_mean['dG [KJ/mol]'], '.k', label='dG-Te [KJ/mol]')
        idx = df_std.index
        df_mean_idx = df_mean.loc[idx]
        plt.fill_between(df_mean_idx['Te[K]'],df_mean_idx['dG [KJ/mol]'] - 2* df_std['dG [KJ/mol]'], df_mean_idx['dG [KJ/mol]'] + 2* df_std['dG [KJ/mol]'], color='g', alpha=0.3, label='Expanded Uncertainty')
        plt.ylim(-200,100)
        plt.xlabel('Temperature - Te [K]')
        plt.ylabel('Change in Gibbs Free Energy [KJ/mol]')
        plt.legend()
        return