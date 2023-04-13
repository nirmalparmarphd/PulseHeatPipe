# PulseHeatPipe
## Pulsating Heat Pipe (PHP): Advanced data analysis and Machine Learning

>Direction to perform data analysis on the PHP experimental data.

MDF - Manually defined functions for Advanced Data Analysis and Machine Learning.

MDF includes various functions for data analysis and for data plotting/visualisation specifically for PHP experimental data.

For more details: help(mdf.function_name), eg: help(mdf.DataETL) or for general help, eg: help(mdf)
    
    1. DataETL
    2. GibbsFE
    3. DataChop
    4. DataArrange
    5. DataProvAvg
    6. BestTP
    7. PlotAllData
    8. PlotTempData
    9. PlotEUTemp
    10. PlotEUPres
    11. PlotEUTR
    12. PlotEUTP
    13. PlotEUGFE
    14. PlotEUdG
    15. PlotEUdT

    HELP: help(mdf.function)

Example:
```
>>> help(mdf.DataETL)
>>> Help on function DataETL in module mdf:

    DataETL(datapath: str)
        DataETL loads experimental data from all experimental data files (.xlsx).
        Filters data and keeps only important columns.
        Combine selected data and save to csv file.
        Conver units to MKS [K, bar] system and save to csv file. 
        
        useage: df, df_conv = DataETL('datapath')
```
**NOTE**: The experimental data file must prepared in '.xlsx' formate. The data must contain atleast following columns with mentioned titles:
[Time (Min), Tc - AVG 1&2 (oC), Te - AVG 1,2,3 (oC), Pressure (mm of Hg), Te - Tc (oC), Q (W), Resistance (oC/W)]

Author: Nirmal Parmar, PhD