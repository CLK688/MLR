import pandas as pd
import numpy as np

datafile = 'data.xlsx'
data = pd.read_excel(datafile)
data = data.iloc[:, :].values 
df = pd.DataFrame(data,columns=['Concentration', 'Temperature', 'Particle_size', 'Enhance'], index=None)
df.to_csv("data.csv")