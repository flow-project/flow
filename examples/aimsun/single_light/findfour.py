# A naive solution to print all combination 
# of 4 elements in A[] with sum equal to X  
def findFourElements(A, n, X): 
      
    # Fix the first element and find  
    # other three 
    for i in range(0,n-3): 
          
        # Fix the second element and  
        # find other two 
        for j in range(i+1,n-2): 
              
            # Fix the third element  
            # and find the fourth 
            for k in range(j+1,n-1): 
                  
                # find the fourth 
                for l in range(k+1,n): 
                      
                    if A[i] + A[j] + A[k] + A[l] == X: 
                        print ("%d, %d, %d, %d"
                        %( A[i], A[j], A[k], A[l])) 
  
# Driver program to test above function 
A = [10, 2, 3, 4, 5, 9, 7, 8] 
n = len(A) 
X = 23
findFourElements (A, n, X) 
  
# This code is contributed by shreyanshi_arun 
