# author: Jelly Lee
# create date: 06/16/2024
# last modification: 06/22/2024
# status: Completed on 06/22/2024

"""
This program applies the thomas algorithm to the matrix appeared in Logistic Process.
A matrix is a tridiagonal matrix, and should look like
1
X  X  X
   X  X  X
      X  X  X
         ........
               X  X  X
                     1
, i.e., A_{1,1} = A_{n,n} = 1, A_{1,2} = A_{n, n-1} = 0, A_{i,i} != 0 for all i  (*)
It ONLY holds when condition (*) is satisfied, and MUST be modified to solve a general triangular matrix problem.
"""

import numpy as np

def tridiagonal_to_upper(A, b):
    """
    1
    X  X  X
       X  X  X
          X  X  X
             ........
                   X  X  X
                         1
    Step 1:
    1
       X  X
       X  X  X
          X  X  X
             ........
                   X  X  X
                         1
    Step 2:
    1
       1  X
       X  X  X
          X  X  X
             ........
                   X  X  X
                         1
    Step 3:
    1
       1  X
          X  X
          X  X  X
             ........
                   X  X  X
                         1
    1
       1  X
          1  X
             X  X
             ........
                   X  X  X
                         1
    ......
    1
       1  X
          1  X
             1  X
             ........
                      X  X
                         1
    Step 4:
    1
       1  X
          1  X
             1  X
             ........
                      1  X
                         1

    Note: numbers on the diagonal MUST be non-zero; condition (*) SHOULD hold
    A_{1,1} = A_{n,n} = 1, A_{1,2} = A_{n, n-1} = 0, A_{i,i} != 0 for all i  (*)
    :param A: -
    :param b: -
    :return: Upper matrix A, and vector b (with the same elementary transformation on A)
    """
    n = len(b)
    for i in range(n-1):
        if i == 0:
            m = A[i+1][i] / A[i][i]
            b[i+1] = b[i+1] - m * b[i]
            A[i + 1][i] = 0
        else:
            # Step 1: standardize
            if A[i][i] != 1:
                A[i][i + 1] = A[i][i + 1] / A[i][i]
                b[i] = b[i] / A[i][i]
                A[i][i] = 1
            if i < n - 2:
                # Step 2: elimination
                m = A[i + 1][i] / A[i][i]
                A[i + 1][i + 1] = A[i + 1][i + 1] - m * A[i][i + 1]
                b[i + 1] = b[i + 1] - m * b[i]
                A[i + 1][i] = 0
    return A, b

def upper_to_diag(A, b):
    '''
    1
       1  X
          1  X
             1  X
             .....
                   1  X
                      1  X
                         1
    Step 1:
    1
       1  X
          1  X
             1  X
             .....
                   1  X
                      1
                         1
    Step 2:
    1
       1  X
          1  X
             1  X
             .....
                   1
                      1
                         1
    Step 3:
    1
       1  X
          1
             1
             .....
                   1
                      1
                         1
    1
       1
          1
             1
             .....
                   1
                      1
                         1
    Note: numbers on the diagonal MUST be 1
    :param A: -
    :param b: -
    :return: Vector b (After the manipulation A is an Identity Matrix)
    '''
    n = len(b)
    for i in range(n-2, 0, -1):
        #n-2, n-1, ..., 1(second row)
        # elimination
        m = A[i][i + 1] / A[i + 1][i + 1]
        b[i] = b[i] - m * b[i + 1]
    return b

if __name__ == '__main__':
    '''The following is a test example to verify that the algorithm works well.'''
    A = np.array([[1, 0, 0, 0],
                  [1, 2, 1, 0],
                  [0, 1, 2, 1],
                  [0, 0, 0, 1]], dtype=float)

    b = np.array([5, 6, 6, 5], dtype=float)
    A, b = tridiagonal_to_upper(A, b)
    print(A)
    print(b)
    b = upper_to_diag(A, b)
    print(b)





