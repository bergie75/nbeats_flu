import pandas as pd
import os
import numpy as np

cwd = os.getcwd()
datafile = os.path.join(cwd, "nbeats_flu", "data", "FluSurveillance_Custom_Download_Data.csv")

# cumulative rate hospitalized is per one hundred thousand
usecols=["YEAR", "WEEK", "AGE CATEGORY", "RACE CATEGORY", "SEX CATEGORY", "VIRUS TYPE CATEGORY", "CUMULATIVE RATE"]
df = pd.read_csv(datafile, usecols=usecols)
overall_data = df[df["AGE CATEGORY"]=="Overall"]
overall_data = overall_data[overall_data["RACE CATEGORY"]=="Overall"]
overall_data = overall_data[overall_data["SEX CATEGORY"]=="Overall"].reset_index(drop=True)
flu_A = overall_data[overall_data["VIRUS TYPE CATEGORY"]=="Influenza A"].reset_index(drop=True)
flu_B = overall_data[overall_data["VIRUS TYPE CATEGORY"]=="Influenza B"].reset_index(drop=True)
both = overall_data[overall_data["VIRUS TYPE CATEGORY"]=="Overall"].reset_index(drop=True)

flu_A_cases = np.array(flu_A["CUMULATIVE RATE"])
flu_B_cases = np.array(flu_B["CUMULATIVE RATE"])
both_cases = np.array(both["CUMULATIVE RATE"])
print(both_cases[30:60])
#print(both.head(50))