#References
#https://stackoverflow.com/
#questions/47330812/find-the-longest-diagonal-of-an-element-in-a-matrix-python

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Create a matrix of euclidean distances between the numeric values of vectors x and y. 
def distanceMatrix(x, y):
   leny=len(y)
   lenx=len(x) 
   distances= [[0] * lenx for i in range(leny)]
   for i in range(len(y)):
     for j in range(len(x)):
        distances[i][j] = float((x[j]-y[i]))**2
   return distances

# Take a matrix as argument to get the longest contiguous match between numeric vectors (rows and columns)
# The matrix is tranformeded to a numpy array. 
# This function uses the start postion of the diagonal with the most zeros and generates the coordinates of zeros
# on that diagonal.  

def indexOfMatchingVal(matrix):   
   start_pos=[0,0]
       # Initiated at position [0,0] of the distance matrix
   maxdiag = 0
   a = np.array(matrix)
      # Changing matrix to numpy arrays
   print a
   for i in range (0, len(a)):
     for j in range (0, len(a[0])):
       k = 0;
       while (i+k < len(a) and j+k < len(a[0]) and a[i+k][j+k] == 0.):
         k+=1;
         if (k > maxdiag):
           maxdiag = k
           start_pos[0] = i
           start_pos[1] = j
   print maxdiag
   x_rg = maxdiag+1
   y_rg = maxdiag+1
   if len(a) == len(a[0]):
      x_rg = x_rg-1
      y_rg = y_rg-1
   #elif len(a) < len(a[0]):
      #x_rg = x_rg
   col = range(start_pos[0],x_rg,1) # Not as robust as it should be. 
   row = range(start_pos[1],y_rg,1)
   if len (row)> maxdiag:
      row = range(start_pos[1],y_rg-1,1)
   elif len(col)> maxdiag:   
      col = range(start_pos[0],x_rg-1,1)
   #if len (row)< maxdiag:
   #   row = range(start_pos[1],y_rg+1,1)
   #elif len(col)< maxdiag:   
   #   col = range(start_pos[1],x_rg+1,1)  
   print start_pos
   print col
   print row
   return row, col   
   
# This function takes two lists of indexes retrieves the corresponding blood glucose values and compare them;
# It will return True is those values are exactly the same. 
def locateDuplicate(listOfIndx1, listOfindx2):
       # List of indexes changed to list changed to numpy array
    ref_index1 = np.array(df1.jsonRowIndex.iloc[x_index]).ravel().tolist()
    ref_index2 = np.array(df2.jsonRowIndex.iloc[y_index]).ravel().tolist()
      # Locate duplicated records from the patient's cgm data file.
    dup_records1= data.loc[data['jsonRowIndex'].isin(ref_index1),['jsonRowIndex','utcTime','value']]
    dup_records2= data.loc[data['jsonRowIndex'].isin(ref_index2),['jsonRowIndex','utcTime','value']]
    uploadBGValues = dup_records1.value.sum() == dup_records2.value.sum()
    #duplicateFrame = pd.concat([dup_records1, dup_records2], axis=1)
    #return duplicateFrame
    return uploadBGValues
 
#==================================================================================================================
#==================================================================================================================
data = pd.read_csv('C:/Users/Demba/Documents/D_File/USF/capstone/data2.csv', delimiter=',')
data.loc[data['type'].isin(['cbg'])]

data2 = pd.read_csv('C:/Users/Demba/Documents/D_File/USF/capstone/data3.csv', delimiter=',')
data2.loc[data['type'].isin(['cbg'])]


# Converting blood glucose values from mmol to mg/DL
data.loc[:,'value'] *= 18.01559 
#data2.loc[:,'value'] *= 18.01559 

# Issue:find a way to get rid of extra row added (NaN values) at the botton the dataframe. 
# Scenario1=  where the records are duplicated at the beginning of both uploads ids and interlaced.

#Load patient file.  Subset 
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


# Changing Blood Glucose values to 1D numpy array.
#v1 = np.array(df1["value"])# array of glucose values for 'upid_3fc32e5ad912a8ea7efced9151804bdb' 
#v2 = np.array(df2["value"])# array of glucose values for 'upid_3c41703c2d3a8b97f479afdb6ccf799f'


#v3 = np.array(df3["value"])# array of glucose values for 'upid_17db2d2a0ae0e02a12c0a5067e5fe85b' 
#v4 = np.array(df4["value"])# array of glucose values for 'upid_5fad608cf32bd03a1cd56e3bb1fdb834'

#v5 = np.array(df5["value"])# array of glucose values for 'upid_' 
#v6 = np.array(df6["value"])# array of glucose values for 'upid_'


# Issue:find a way to get rid of extra row added (NaN values) at the botton the dataframe. Temporary solution is to
# slice it to get rid of the row added. 

#x = v1[:168]
#y = v2[:807]

#x = v3[:288]
#y = v4[:1000]

#x = v5[:168]
#y = v6[:807]

############Test
#x=  [1,1,2,3,2,0,1]
#y=[0,1,1,2,3,2,0]

#x=  [1,1,2,3,2,0,1]
#y=[0,1,1,2,3,2,1,1]

# Not working
#x =    [1,0,1,2,3,2,0,1,0,1,2,1]
#y =   [0,1,1,2,3,2,1,1]


x =    [1,0,1,2,3,2,0,1,0,1]
y =   [   0,1,2,3,2,1,1]

#square matrix 
#x=[1,1,2,3,2,0]
#y=[1,1,2,3,2,0]



#Blood Glucose values from upload ids upid_3fc32e5ad912a8ea7efced9151804bdb & upid_3c41703c2d3a8b97f479afdb6ccf799f
#dup_bgval_upload1 = df1.value.iloc[x_index]
#dup_bgval_upload2 = df2.value.iloc[y_index]
# Values returned from the matching_idx output are the same # True
#print dup_bgval_upload1.sum() == dup_bgval_upload2.sum()

# Matching values' position  mirrored back to the patient's file dataframe


# Access jsonRowIndex used as reference indexes to locate duplicated records on patient's file
print"--###------------------------------------------------------------------------------------"

print"------------###------------------###-----------------###-------------------###-----------"



print"-----------------------------------------------------------------------------------------"
listMatchingIndx = distanceMatrix(x,y)
x_index, y_index = indexOfMatchingVal(listMatchingIndx)
print locateDuplicate (x_index, y_index)

# Visualization of BG values for subsets based on the upload ids 
#plt.subplot(2, 1, 1)
#plt.plot(dup_bgval_upload1,'g-')
#plt.ylabel('dup_bgval_upload1')
#plt.subplot(2, 1, 2)
#plt.plot(dup_bgval_upload2,)
#plt.ylabel('dup_bgval_upload2')
###
#plt.show()