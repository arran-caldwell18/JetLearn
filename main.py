import numpy as np

#List 1-10 then 2 numpy aray
print()
numpy_list  = np.array([1,2,3,4,5,6,7,8,9,10])
normal_list = [1,2,3,4,5,6,7,8,9,10]

print("Numpy",numpy_list)
print("Normal",normal_list)


#Multiply them both by 2
print()
numpy_list_2  = normal_list * 2
normal_list_2 = numpy_list * 2

print("Numpy",numpy_list_2)     #   doubles list (by 2 lists in 1)
print("Normal",normal_list_2)    #   doubles individual values in list


#Add Both lists
print()
numpy_list_addition = numpy_list + numpy_list
normal_list_addition = normal_list + normal_list

print("Numpy",numpy_list_addition) # Adds lists individual values
print("Normal",normal_list_addition) #Adds extra list to the end


######################################################################