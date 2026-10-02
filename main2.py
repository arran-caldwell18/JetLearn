import numpy as np

array1 = np.zeros((4,5))   
array2 = np.ones((3,3))     
array3 = np.arange(20).reshape(4,5)

array_list =  [array1 , array2 , array3]

#Loop over numpy array
for index , arr in enumerate(array_list):
    
    #Print -> dims , shape , element total , mem usage

    print(f"{index+1}:")
    print(f"    Dimensions : {np.ndim(arr)}")
    print(f"    Shape      : {np.shape(arr)}")
    print(f"    Size       : {np.size(arr)}")
    print(f"    Mem Usage  : {arr.nbytes}")        

#My pred
    #arr1 -> 20 * 8 bytes (assuming 8bit integers)
    #arr2 -> 9  * 8 bytes (assuming 8bit integers)
    #arr3 -> 20 * 8 bytes (assuming 8bit integers)

#Real Answer:
    #Inegers are dynamically sized -> BUT in this case I am Correct as All Integers in given arrays are 8Bytes