import math
import numpy as np

def generate(x, y) -> list:
    
    num_nodes = math.ceil(x * y / 2)

    adj_matrix = [[0 for n in range(0,num_nodes+1)] for m in range(0,num_nodes+1)]
    adj_matrix = np.array(adj_matrix)
    for n in range(0,num_nodes+1):
        adj_matrix[0][n] = n
        adj_matrix[n][0] = n
    
    long = math.ceil(x/2)
    short = long - 1
    
    # Used for testing
    """
    print(short)
    print(long)
    print(num_nodes)
    """
    
    # r stands for repetitions
    # r is incremented every 4th row (not counting row 1)
    # in other words, after every sequence of WW,||,MM,||
    # j is what row of the lattice you are on
    
    j = 1
    r = 0
    
    if x % 2 == 1:  #odd x
        while True:
            for i in range(1+(2*(x*r)),long+1+(2*(x*r))):   #WWW
                if i > 1+(2*(x*r)):   #/
                    adj_matrix[i][i+short] = 1
                    adj_matrix[i+short][i] = 1
                if i < long+1+(2*(x*r))-1:    #\
                    adj_matrix[i][i+long] = 1
                    adj_matrix[i+long][i] = 1
            
            j += 1
            if j >= y:
                break
            
            for i in range(x+1+(x*2*r),x+1+(x*2*r)+short):   #|| short
                adj_matrix[i][i-short] = 1
                adj_matrix[i-short][i] = 1
            
            j += 1
            if j >= y:
                break
                    
            for i in range(x+long+(x*(r*2)), x+long+(x*(r*2))+long):    #MMM
                if i < x+long+(x*(r*2))+long-1:   #/
                    adj_matrix[i][i-short] = 1
                    adj_matrix[i-short][i] = 1    
                if i > x+long+(x*(r*2)):   #\
                    adj_matrix[i][i-long] = 1
                    adj_matrix[i-long][i] = 1
            
            j += 1
            if j >= y:
                break
            
            for i in range((x*2*(r+1)+1),((x*2*(r+1)+1+long))):    #(x*2+1+r,x*2+long+1+r):   #|||
                adj_matrix[i][i-long] = 1
                adj_matrix[i-long][i] = 1
            
            j += 1
            if j >= y:
                break
            r += 1
    
    else:   #even x
        while True:
            for i in range(1+(2*(x*r)),long+1+(2*(x*r))):   #WWW
                if i > 1+(2*(x*r)):   #/
                    adj_matrix[i][i+short] = 1
                    adj_matrix[i+short][i] = 1
                if i < long+1+(2*(x*r)):    #\
                    adj_matrix[i][i+long] = 1
                    adj_matrix[i+long][i] = 1
            
            j += 1
            if j >= y:
                break
                
            for i in range(x+1+(x*2*r),x+1+(x*2*r)+long):   #|| short
                adj_matrix[i][i-long] = 1
                adj_matrix[i-long][i] = 1
                
            j += 1
            if j >= y:
                break
                        
            for i in range(x+long+(x*(r*2)), x+long+(x*(r*2))+long+1):    #MMM
                if i < x+long+(x*(r*2))+long+1:   #/
                    adj_matrix[i][i-long] = 1
                    adj_matrix[i-long][i] = 1    
                if i > x+long+(x*(r*2)+1):   #\
                    adj_matrix[i][i-long-1] = 1
                    adj_matrix[i-long-1][i] = 1
            
            j += 1
            if j >= y:
                break
            
            for i in range(x+1+(x*(r+1)),x+1+(x*(r+1)+long)):   #|| short
                adj_matrix[i][i-long] = 1
                adj_matrix[i-long][i] = 1
            
            j += 1
            if j >= y:
                break
            
            r += 1
    
        
    return adj_matrix


# Output to console
# print(generate(3,4))


#Output to a new file
# with open('adj_matrix_generator_output.txt', 'w') as f:
#     for n in adj_matrix.generate(3,4):
#         j = len(n)-1
#         i = 0
#         for m in n:
#             f.write(str(m))
#             if i < j:
#                 f.write(",")
#             i += 1
#         f.write("\b\n")
    
