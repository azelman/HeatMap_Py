import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

dates = pd.date_range('04/01/2015', periods=2000, freq='4H')
df = pd.DataFrame(0, index=dates, columns=['EUR', 'USD', 'GBP', 'CAD', 'AUD', 'NZD', 'CHF', 'JPY'])

def update(x, first, second):
  df[first][str(x.DATE).replace('.','-') + ' ' + str(x.TIME)] +=  x["UP2"] + x["UP1"] - x["DOWN1"] - x["DOWN2"]
  df[second][str(x.DATE).replace('.','-') + ' ' + str(x.TIME)] +=  x["DOWN1"] + x["DOWN2"] - x["UP2"] - x["UP1"]
  
def updateData(first, second):
  print 'Loading Currency Data ' + './' + first + second + '.txt'
  pair0 = pd.read_csv('./' + first + second +'.txt', sep=';')
  pair0 = pair0[(pair0.TIME == '00:00:00') | (pair0.TIME == '04:00:00') | (pair0.TIME == '08:00:00') | (pair0.TIME == '12:00:00') | (pair0.TIME == '16:00:00') | (pair0.TIME == '20:00:00')]
  pair1 = pair0.shift()
  pair0['UP2'] = (pair0["CLOSE"] > pair1["HIGH"]).astype(int)
  pair0['UP1'] = (pair0["CLOSE"] > pair1["CLOSE"]).astype(int)
  pair0['DOWN1'] = (pair0["CLOSE"] < pair1["CLOSE"]).astype(int)
  pair0['DOWN2'] = (pair0["CLOSE"] < pair1["LOW"]).astype(int)  
  pair0 = pair0.drop(pair0.index[:1])
  #	print 'Updating global heatmap with ' + first + second
  pair0.apply(update,axis=1,first=first, second=second)
  
def loadData():
  updateData('EUR', 'USD')
  updateData('EUR', 'GBP')
  updateData('EUR', 'AUD')
  updateData('EUR', 'CAD')
  updateData('EUR', 'NZD')
  updateData('EUR', 'CHF')
  updateData('EUR', 'JPY')
  
  updateData('GBP', 'AUD')
  updateData('GBP', 'CAD')
  updateData('GBP', 'CHF')
  updateData('GBP', 'JPY')
  updateData('GBP', 'NZD')
  updateData('GBP', 'USD')
  
  updateData('AUD', 'CAD')
  updateData('AUD', 'CHF')
  updateData('AUD', 'JPY')
  updateData('AUD', 'NZD')
  updateData('AUD', 'USD')
  
  updateData('NZD', 'CAD')
  updateData('NZD', 'CHF')
  updateData('NZD', 'JPY')
  updateData('NZD', 'USD')
  
  updateData('USD', 'CAD')
  updateData('USD', 'CHF')
  updateData('USD', 'JPY')

  updateData('CAD', 'CHF')
  updateData('CAD', 'JPY')
  
  updateData('CHF', 'JPY')
  global df
  df = df[(df.EUR != 0) & (df.USD != 0) & (df.AUD != 0)]
  df.to_csv('heatmap.csv')
  print len(df)

def main():
  loadData()
  print 'Done'

if __name__ == '__main__':
  main()