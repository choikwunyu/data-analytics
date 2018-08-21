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
   print a
   for i in range (0, len(a)):
     for j in range (0, len(a[0])):
       k = 0;
       while (i+k < len(a) and j+k < len(a[0]) and a[i+k][j+k] == 0):
         k+=1;
         if (k > maxdiag):
           maxdiag = k
           start_pos[0] = j
           start_pos[1] = i
   print start_pos
   print maxdiag
   size = maxdiag+1
   if start_pos[0] != 0 and start_pos[1] != 0:
      size= maxdiag+2
      print"was I here"
   if (start_pos[0] != 0 and start_pos[1] != 0) and start_pos[0] == start_pos[1]:
      size= maxdiag+1
      print"I was here"
   if len(a) == len(a[0]) and len(a[0]) == maxdiag:
     size = maxdiag
     print"I was here not there"
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

data = pd.read_csv('C:/Users/Demba/Documents/D_File/USF/capstone/data2.csv', delimiter=',')
data.loc[data['type'].isin(['cbg'])]
#Converting blood glucose values from mmol to mg/DL
data.loc[:,'value'] *= 18.01559 


#data2 = pd.read_csv('C:/Users/Demba/Documents/D_File/USF/capstone/data3.csv', delimiter=',')
#data2.loc[data['type'].isin(['cbg'])] 
##data2.loc[:,'value'] *= 18.01559 
#
## Issue:find a way to get rid of extra row added (NaN values) at the botton the dataframe. 
## Scenario1=  where the records are duplicated at the beginning of both uploads ids and interlaced.
#
##Load patient file.  Subset?????

df1 = data.loc[data['uploadId'] =='upid_3fc32e5ad912a8ea7efced9151804bdb',['jsonRowIndex','utcTime','value']]
df2 = data.loc[data['uploadId'] =='upid_3c41703c2d3a8b97f479afdb6ccf799f',['jsonRowIndex','utcTime','value']]

#Scenario2 

#df3 = data.loc[data['uploadId'] =='upid_17db2d2a0ae0e02a12c0a5067e5fe85b', ['jsonRowIndex','utcTime','value']]
#df4 = data.loc[data['uploadId'] =='upid_5fad608cf32bd03a1cd56e3bb1fdb834', ['jsonRowIndex','utcTime','value']]

#df3 = data.loc[data['uploadId'] =='upid_5fad608cf32bd03a1cd56e3bb1fdb834', ['jsonRowIndex','utcTime','value']]
#df4 = data.loc[data['uploadId'] =='upid_830c6de3e2ecbbec6fbad0cecc64bdf5', ['jsonRowIndex','utcTime','value']]


# Scenario3 

#df5 = data2.loc[data2['uploadId'] ==' ', ['jsonRowIndex','utcTime','value']]
#df6 = data2.loc[data2['uploadId'] ==' ', ['jsonRowIndex','utcTime','value']]


# Changing Blood Glucose values to 1D numpy array.????

v1 = np.array(df1["value"])# array of glucose values for 'upid_3fc32e5ad912a8ea7efced9151804bdb' 
v2 = np.array(df2["value"])# array of glucose values for 'upid_3c41703c2d3a8b97f479afdb6ccf799f'


#v3 = np.array(df3["value"])# array of glucose values for 'upid_5fad608cf32bd03a1cd56e3bb1fdb834' 
#v4 = np.array(df4["value"])# array of glucose values for 'upid_17db2d2a0ae0e02a12c0a5067e5fe85b'

#v5 = np.array(df5["value"])# array of glucose values for 'upid_' 
#v6 = np.array(df6["value"])# array of glucose values for 'upid_'


# Issue:find a way to get rid of extra row added (NaN values) at the botton the dataframe. Temporary solution is to
# slice it to get rid of the extra row. 

x = v1[:167]
y = v2[:807]

#x = v1
#y = v2
#

#x = v3[:288]
#y = v4[:1000]

#x = v5[:168]
#y = v6[:807]

############Test
#x=  [1,1,2,3,2,0,1]
#y= [0,1,1,2,3,2,0,1]

#x =    [0,1,2,3,2,0,1]
#y =  [0,1,1,2,3,2,1,1]

# Not working. Noticed that here the coordinate does not start at either column 0 or row 0. 
#x =  [1,1,1,2,3,2,0,1,0,1,2,1,2] # we have to account for length diff.
#y =  [0,1,1,2,3,2,1,1,0] #??????????

#x =[1,0,1,2,3,2,0,1,0,1]
#y =  [0,1,2,3,2,1,1]

#square matrix 
#x=[1,1,2,3,2,0]
#y=[1,1,2,3,2,0]

#x=        [1,1,2,3,2,0,1]
#y=     [0,1,1,2,3,2,0]


##############Main################4

matrix = distanceMatrix(x, y)
a = np.array(matrix)
initial_pos, size= diagonalStartingIndex(a)
coordinates = (positionOfdiagonalZeros(initial_pos, size))
dup_records1,dup_records2, Bool, dup_bgval_upload1, dup_bgval_upload2  =locateDuplicateValues(coordinates)

print dup_records1
print dup_records1
print dup_bgval_upload1
print dup_bgval_upload2
print Bool

#Visualization of BG values for subsets based on the upload ids 
plt.subplot(2, 1, 1)
plt.plot(dup_bgval_upload1,'g-')
plt.ylabel('dup_bgval_upload1')
plt.subplot(2, 1, 2)
plt.plot(dup_bgval_upload2,)
plt.ylabel('dup_bgval_upload2')
plt.show()