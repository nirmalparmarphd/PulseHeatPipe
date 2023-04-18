# PulseHeatPipe

>Direction to perform data analysis on the PHP experimental data.

Module for data analysis and for data plotting/visualisation specifically for PHP experimental data.

## PulseHeatPipe - Module for Advanced Data Analysis and Machine Learning.

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

Example:
```
# importing module
from analysis import PulseHeatPipe
from analysis import DataVisualisation

analysis = PulseHeatPipe("data/al2o3_diwater_exp/60_FR/")
visual = DataVisualisation('Al2O3_DI_60FR')

# calling help
help(analysis.data_etl)
help(visual.plot_all_data)

# using methods eg;
df, df_conv = analysis.data_etl()
visual.plot_all_data(df_gfe)

```
**NOTE**: The experimental data file must prepared in '.xlsx' formate. The data must contain atleast following columns with mentioned titles:

**Data.xlsx format**

| 'Time (Min)' | 'Tc - AVG (oC) | 'Te - AVG (oC)' | 'Pressure (mm of Hg)' | 'Te - Tc (oC)' | 'Q (W)' |'Resistance (oC/W)' |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 35 | 700 | 5 | 80 | 0.06 |
| --- | --- | --- | --- | --- | --- | --- |