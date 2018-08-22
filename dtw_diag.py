#References
#https://stackoverflow.com/
#questions/47330812/find-the-longest-diagonal-of-an-element-in-a-matrix-python
#https://codereview.stackexchange.com/questions/146935/find-diagonal-positions-for-bishop-movement
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt

# Create a matrix of euclidean distances between the numeric values of vectors x and y.
#Args:
# x = vector1 and y = vector. vectors are blood glucose values from two different upload events.   
def distanceMatrix(x, y):
   leny=len(y)
   lenx=len(x) 
   distances= [[0] * lenx for i in range(leny)]
   for i in range(len(y)):
     for j in range(len(x)):
        distances[i][j] = (x[j]-y[i])**2
   return distances


# Take a matrix as argument to get the longest contiguous match between numeric vectors (rows and columns)
# The matrix is tranformed to a numpy array. 
# This function uses the start postion of the diagonal with the most zeros and generates the coordinates of zeros
# on that diagonal.  
def diagonalStartingIndex(matrix):   
   start_pos=[0,0]
       # Initiated at position [0,0] of the distance matrix
   maxdiag = 0
   a = np.array(matrix)
      # Changing matrix to numpy arrays
   for i in range (0, len(a)):
     for j in range (0, len(a[0])):
       k = 0;
       while (i+k < len(a) and j+k < len(a[0]) and a[i+k][j+k] == 0):
         k+=1;
         if (k > maxdiag):
           maxdiag = k
           start_pos[0] = j
           start_pos[1] = i
   size = maxdiag+1
   if start_pos[0] != 0 and start_pos[1] != 0:
      size= maxdiag+2
   if (start_pos[0] != 0 and start_pos[1] != 0) and start_pos[0] == start_pos[1]:
      size= maxdiag+1
   if len(a) == len(a[0]) and len(a[0]) == maxdiag:
     size = maxdiag
   return start_pos, size

# This function generates a list of tuples of the positons of the diagonal values of interest
#in a matrix given a starting position. The value of interest here is zero. 
#Args: 
# coord = starting coordinates of the diagnoal with the longest occurnece of zeros.
# size  = The number of zeors on the diagoanl.

def positionOfdiagonalZeros (coord, size):
    x, y = coord
    list_coordinate = list(chain(
        [((x), (y))],
        zip(range(x + 1, size, 1), range(y + 1, size, 1)),
    ))
    return  list_coordinate

# This function takes a list of tuples of indexes, with each indexes
# It will return True is those values are exactly the same. 
# Locate duplicated records from the patient's cgm data file.
def locateDuplicateValues(tuple_idx):
      x_index = ([x[0] for x in tuple_idx]);y_index = ([x[1] for x in tuple_idx])
      ref_index1 = df1.jsonRowIndex.iloc[x_index]
      ref_index2 = df2.jsonRowIndex.iloc[y_index]
      dup_records1 = data.loc[data['jsonRowIndex'].isin(ref_index1),['jsonRowIndex','utcTime','value']]
      dup_bgval_upload1 = dup_records1.value
      dup_records2= data.loc[data['jsonRowIndex'].isin(ref_index2),['jsonRowIndex','utcTime','value']]
      dup_bgval_upload2 = dup_records2.value
      uploadBGValues = dup_records1.value.sum() == dup_records2.value.sum()
      return dup_records1, dup_records2, uploadBGValues,dup_bgval_upload1,dup_bgval_upload2
      
#==================================================================================================================
#==================================================================================================================


### Main###

data = pd.read_csv('C:/Users/Demba/Documents/D_File/USF/capstone/data2.csv', delimiter=',')
data.loc[data['type'].isin(['cbg'])]
#Converting blood glucose values from mmol to mg/DL
data.loc[:,'value'] *= 18.01559 

# Geenrating two subsets of the patient files based on two uploads events
df1 = data.loc[data['uploadId'] =='upid_3fc32e5ad912a8ea7efced9151804bdb',['jsonRowIndex','utcTime','value']]
df2 = data.loc[data['uploadId'] =='upid_3c41703c2d3a8b97f479afdb6ccf799f',['jsonRowIndex','utcTime','value']]

# Changing Blood Glucose values to 1D numpy array.????
v1 = np.array(df1["value"])# array of glucose values for 'upid_3fc32e5ad912a8ea7efced9151804bdb' 
v2 = np.array(df2["value"])# array of glucose values for 'upid_3c41703c2d3a8b97f479afdb6ccf799f'

x = v1[:167]
y = v2[:807]

matrix = distanceMatrix(x, y)
a = np.array(matrix)
initial_pos, size= diagonalStartingIndex(a)
coordinates = (positionOfdiagonalZeros(initial_pos, size))
dup_records1,dup_records2, Bool, dup_bgval_upload1, dup_bgval_upload2  =locateDuplicateValues(coordinates)
print Bool
#Visualization of BG values for subsets based on the upload ids 
plt.subplot(2, 1, 1)
plt.plot(dup_bgval_upload1,'g-')
plt.ylabel('dup_bgval_upload1')
plt.subplot(2, 1, 2)
plt.plot(dup_bgval_upload2,)
plt.ylabel('dup_bgval_upload2')
plt.show()