import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

fx = {}

def getFXrecord(symbol, date):
  global fx
  try:
    record = fx[symbol][date]
    return fx[symbol][date]
  except KeyError:
    pair = pd.read_csv('./' + symbol +'.txt', sep=';')

  pairHash = {}   
  for record in pair.values:    
    key = record[0] + ' ' + record[1]
    pairHash[key] = record
    
  fx[symbol] = pairHash
  return fx[symbol][date]    
    
    
def getFutureForexGain(x):
  stopLoss = 0.4
  
  first = x.predictMAX
  second = x.predictMIN
  factor = 1
  if first == 'JPY' or second == 'JPY':
    factor *= 0.01
  
  if first == second:
    return 0
  
  try:
    record = getFXrecord(first + second, x.name.replace('-','.'))
  except IOError:
    record = getFXrecord(second + first, x.name.replace('-','.'))
    factor *= -1

  #pair['NEWDATE'] = pair.apply(lambda x: str(x.DATE).replace('.','-') + ' ' + str(x.TIME), axis=1)

  #record = pair[(pair.NEWDATE == x.name).shift().fillna(value=False)]
  #[2    3   4     5]
  #HIGH;LOW;CLOSE;OPEN
  
  gain = ((record[4] - record[5]) * factor)

  if factor > 0 and ((record[5] - record[3]) > stopLoss):
    gain = -1*stopLoss*factor
  if factor < 0 and ((record[2] - record[5]) > stopLoss):
    gain = -1*stopLoss*factor
  return gain if gain > -1*stopLoss else -1*stopLoss

def main():
  df = pd.read_csv('./heatmap.csv', index_col=0)
  
  df['MAX'] = df.ix[:,'EUR':'JPY'].idxmax(axis=1)
  df['MIN'] = df.ix[:,'EUR':'JPY'].idxmin(axis=1)
  
  df['nextMAX'] = df.ix[:,'EUR':'JPY'].idxmax(axis=1).shift(-1)
  df['nextMIN'] = df.ix[:,'EUR':'JPY'].idxmin(axis=1).shift(-1)
  
  FX = list(enumerate(np.unique(df['nextMAX'])))    # determine all values of nextMAX,
  FX_dict = { nextMAX : i for i, nextMAX in FX }              # set up a dictionary in the form  Ports : index
  FX_dict_r = { i : nextMAX for i, nextMAX in FX }
  df['nextMAX'] = df.nextMAX.map( lambda x: FX_dict[x]).astype(int)     # Convert all Embark strings to int
  df['nextMIN'] = df.nextMIN.map( lambda x: FX_dict[x]).astype(int)     # Convert all Embark strings to int
  
  df = df[df.EUR != 0]
  train_df = df[0:600]
  train_df = train_df.drop(['MIN','MAX'],axis=1)
  train_data = train_df.values
  
  #test_df = df[600::]
  #test_df = test_df.drop(['MIN','MAX'],axis=1)
  #test_data = test_df.values
  
  print 'Training...'
  forest_max = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1, max_features='sqrt')
  forest_max = forest_max.fit( train_data[0::,0:8], train_data[0::,8] )
  forest_min = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1, max_features='sqrt')
  forest_min = forest_min.fit( train_data[0::,0:8], train_data[0::,9] )

  print 'Predicting...'
  output_max = forest_max.predict(df.drop(['MIN','MAX'],axis=1).values[0::,0:8]).astype(int)
  output_min = forest_min.predict(df.drop(['MIN','MAX'],axis=1).values[0::,0:8]).astype(int)
  
  df['predictMAX'] = 0
  df['predictMIN'] = 0
  
  output_max = output_max.tolist() 
  output_max.insert(0,1)
  output_max.pop();
  output_min = output_min.tolist() 
  output_min.insert(0,1)
  output_min.pop();

  df['predictMAX'] = output_max
  df['predictMIN'] = output_min    
  
  df.predictMAX = df['predictMAX'].apply(lambda x : FX_dict_r.get(x))
  df.predictMIN = df['predictMIN'].apply(lambda x : FX_dict_r.get(x))
  
  print 'Computing gain...'
  df['GAIN'] = df.apply(getFutureForexGain, axis=1)

  print df[df.EUR != 0]
  print df.GAIN.sum()
  
  #plt.plot(df.GAIN[600::].cumsum()) 
  plt.plot(df.GAIN[0::].cumsum()) 
  plt.show()
  
if __name__ == '__main__':
  main()