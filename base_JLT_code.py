import math
import random
import sys
import numpy


#Generate n points for a d dimensional matrix.
# Bound the values to the given range of min to max (inclusive)
def Generate_Matrix(n, d, minVal=0, maxVal=5000):
	data = []

	#Generate values for each point
	for i in range(n):
		point = []

		#generate values for each dimension/column
		for j in range(d):
			point.append(random.randint(minVal, maxVal))

		data.append(point)


	return data



#Calculate the linear distance between two given d-dimensional points
# Points A and B should be python lists of their values for each dimension
def Calculate_Distance(pointA, pointB):
	d = len(pointA)
	assert(len(pointA) == len(pointB))

	distance = 0
	for i in range(d):
		temp = pointA[i] - pointB[i]
		distance += temp * temp

	return math.sqrt(distance)



#List the k nearest neighbors around the given point in the given data set
# x should be a python list with the correct number of dimensions to fit in data
def KNN_Search(k, data, x):
	assert(len(x) == len(data[0]))

	neighbors = [(sys.maxsize, -1)] #Place holder value, to be replaced

	#We will have to compare x to every point in data
	for i in range(len(data)):
		yDist = Calculate_Distance(x, data[i])

		#Insert in (distance, index) pairs
		if yDist < max(neighbors):
			neighbors.append((yDist, i))

			#Remove the largest if we have k neighbors already
			if len(neighbors) >= k:
				neighbors.remove(max(neighbors))


#List the difference in distances between the two matrices of data
# to the given point/equivalent point
def Calculate_JLT_Distance_Differences(original, transformed, eps):
	assert(len(original) == len(transformed))

	m = len(original)

	results = []
	outOfBounds = []

	for j in range(m):
		point = original[j]
		transformedPoint = transformed[j]

		for i in range(m):
			distOrig = Calculate_Distance(point, original[i])
			distTran = Calculate_Distance(transformedPoint, transformed[i])

			distDiff = distOrig - distTran
			if distDiff < 0:
				distDiff = -distDiff

			results.append(distDiff)

			#Check if in bounds?
			if not ((1-eps)*distOrig <= distTran) or not (distTran <= (1+eps)*distOrig):
				#if len(outOfBounds) < 3:
				#	print(distOrig)
				#	print(point)
				#	print(original[i])

				#	print(distTran)
				#	print(transformedPoint)
				#	print(transformed[i])

				#if not (distTran - distOrig <= eps * distOrig)
				outOfBounds.append( (distDiff, i) )

	#Expecting 2/n^2 values to be out of bounds
	print("TODO 1-1/m = P any points out of bounds")
	expectedNumWrong = 1 - (1/m)
		#2/(len(original) * len(original)) * math.log(len(original))
	if len(outOfBounds) >= expectedNumWrong:
		print("WARNING: More than expected", expectedNumWrong, "out of expected bounds.")
		print("  Got", len(outOfBounds), " distances outside of expected bounds")

	return (outOfBounds, results)



#Script a bunch of tests w/ generated or imported matrix


#Generate a random high-dimensional matrix
N = 500 #Original dimensionality
#m = 100
m=10

originalData = Generate_Matrix(m, N)


#Random transformation mappings:
# These serve as the linear map from data (N dimensional) to projection (d dimensional)
# This mapping f will have the dimensions (N x d)
# Note whether d satisfies the lemma, being larger than 8 ln(m)/eps^2

#eps = 0.5 #Represents out tolerance value, must be in (0,1)
eps = 0.9
assert(eps > 0 and eps < 1)

d = int(8 * math.log(m) / (eps * eps) + 1) #Round up from this low value

if(N < d):
	print("ERROR: Original dimensionality ", d, " is not big enough for ", m, "points")


#Transformation matrix
#N rows and d columns, for (N-dim, m rows) to (d-dim, m rows)
f = Generate_Matrix(N, d, 0, 1)
print("TODO: Need to figure out howto properly generate transformation matrix")
#lipschitz, 1/sqrt()

#Transformed Matrix
transformed = numpy.dot(originalData, f)/math.sqrt(d)

#assert(len(originalData[0]) == N) #Ensure correct actual dimensions
assert(len(transformed[0]) == d)


#Display the two generated matrices
#print(numpy.array(originalData)[:10,:5])
#print(numpy.array(transformed)[:10,:5])

print(d, "= d for m =",m)

print("TODO: Repeat experiment with more transformation matrices, pick best/least dist?")

#Test distances
outOfBounds, results = Calculate_JLT_Distance_Differences(originalData, transformed, eps)

print(len(outOfBounds))

