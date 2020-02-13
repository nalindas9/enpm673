#Import scientific computing package "numpy" 
import numpy as np
from numpy import linalg as la 
np.set_printoptions(suppress=True)

#Insert the A matrix elements here, in this case - the given point correspondences
x1 = 5
x2 = 150
x3 = 150
x4 = 5
y1 = 5
y2 = 5
y3 = 150
y4 = 150
xp1 = 100
xp2 = 200
xp3 = 220
xp4 = 100
yp1 = 100
yp2 = 80
yp3 = 80
yp4 = 200

# Initializing the A matrix 
A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1], [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1], [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2], [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2], [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3], [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3], [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4], [0 , 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])

print 'The given A matrix is: '
print A 
print ' '
# Taking transpose of A matrix
At = np.transpose(A)
print 'The Transpose of the A matrix (At) is: '
print At

# Product of A and A transpose (At)
A1 = np.dot(A,At)
print 'The value of A*At is: ' 
print A1

print ' '
# Eigen values and vectors of A*At
ei1, eiv1 = la.eig(A1) 
eiv1 = np.array(eiv1)

#Sorting the Eigen values in descending order and getting the indices
id1 = np.argsort(-ei1, axis = -1)

print 'Eigen Values for AAt are:' 
print ei1 
print '                                                                                         '
print 'Sorted Eigen values for AAt are: '
print ei1[id1]
print ' '
print 'Eigen Vectors for AAt are:' 
print eiv1
print 'Sorted Eigen vectors for AAt are: '
print eiv1[:,id1]
print ' '
print '*****************************************************************************************'
# Product of At and A 
A2 = np.dot(At,A)
print 'The value of At*A is: ' 
print A2
print ' '
# Eigen values and eigen vectors of At*A
ei2, eiv2 = la.eig(A2)
eiv2 = np.array(eiv2)

#Sorting the Eigen values in descending order and getting the indices
id2 = np.argsort(-ei2, axis = -1)

print 'Eigen Values for AtA are:' 
print ei2
print '                                                                                         '
print 'Sorted Eigen values for AtA are: '
print ei2[id2]
print ' '
print 'Eigen Vectors for AtA are:' 
print eiv2
print 'Sorted Eigen vectors for AtA are: '
print eiv2[:,id2]
print ' '

# Transpose of Eigen vectors of A*At and At*A as U and V are matrix of column vectors
U = eiv1[:,id1]
V = eiv2[:,id2]
print 'U is: ' 
print eiv1
print ' '
print 'V is: '
print eiv2

# Value of sigma obtained by taking diagonal matrix of square roots of eigen values of A*At
sigma = np.array(np.diag(np.sqrt(ei1[id1])))
zero_column = np.array([0,0,0,0,0,0,0,0])
sigma = np.column_stack((sigma, zero_column))
print 'Sigma is: ' 
print sigma
print ' '

# U, Sigma and V computed from the numpy SVD function to compare with obtained values
u, s, vh = np.linalg.svd(A)
print 'U from SVD Function is: ' 
print u
print ' '
print 'Sigma from SVD function is: ' 
print s
print ' '
print 'V from SVD Function is: ' 
print (np.transpose(np.array(vh)))

#The value of x for which Ax=0 is the Eigen vector corresponding to the eigen value = 0 of At*A
# X is the last column of the sorted Eigen vector matrix for At*A, since it corresponds to eigen value = 0
x = np.array(eiv2[:,id2])[:,8]
print 'The Value of x obtained is: '
print x
print ' '
# To confirm the solution of Ax = 0, multiplying A and X to check if 0
print 'Value of A*X is :'
print np.matmul(A, x)

# H is the Homography matrix obtained from x
h1 = x[0:3]
h2 = x[3:6]
h3 = x[6:9]

H = np.transpose(np.column_stack((np.column_stack((h1, h2)), h3)))
print 'The homography matrix H is: '
print ' '
print H

