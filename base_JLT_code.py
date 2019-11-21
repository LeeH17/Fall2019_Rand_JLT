import random
import sys
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances



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

def Calculate_Minimum_JLT_Dimensions(numPoints, epsilon):
	return np.log(numPoints) / (epsilon*epsilon)

def Generate_JLT_Transform_Matrix(k, d, distrType="normal", mean=0, stdDev=1):
	distrType = distrType.lower()

	transform = np.zeros((k,d)) #The new transformation matrix

	#Generate values for each point
	for i in range(1, k):
		#point = []

		#if i % 1000 == 0:
		#	print("at",i)

		#generate values for each dimension/column
		for j in range(1, d):

			#Use a normal/gaussian distribution for internal values,
			# as defined in text
			if distrType == "normal" or distrType == "gaussian":
				transform[i][j] = np.random.normal(mean, stdDev)

			elif distrType == "achioloptas" or distrType == "paper":
				#Simpler version also used in paper
				temp = random.random()
				if temp < 1/6:
					point[i][j] = float(1)
				elif temp > 5/6:
					point[i][j] = float(-1)
				#else leave point at 0


			else:
				print("ERROR: Unknown distribution type provided for generating "
					+ "our transformation matrix! Was passed " + distrType)
				exit()

		#transform.append(point)

	#Re-scale the transformation matrix


	#actually, taken care of already i original transform matrix
	#if distrType == "achioloptas" or distrType == "paper":
	#	#The value as defined in the paper
	#	transform = transform * np.sqrt(3)
	#else:
	#	if distrType != "normal" and distrType != "gaussian":
	#		print("Warning: Undefined distribution type", distrType,
	#			", defaulting to gaussian.")
	#	transform[i][j] = np.random.normal(mean, stdDev)

	return transform

def TransformMatrix(original, JLT_matrix):
	#print(original.data.nbytes, JLT_matrix.data.nbytes, original.shape, JLT_matrix.shape)
	#result = np.dot(original, JLT_matrix)
	result = original.dot(JLT_matrix)

	# sqrt(d/k); d = original dimensionality, k = target dimensionality
	transformationFactor = len(JLT_matrix[0])#len(JLT_matrix) / len(JLT_matrix[0])
	transformationFactor = 1/np.sqrt(transformationFactor)
	#print(len(JLT_matrix[0]), transformationFactor)

	result = result * transformationFactor

	return result



#Calculate the linear distance between two given d-dimensional points
# Points A and B should be python lists of their values for each dimension
def Calculate_Distance(pointA, pointB):
	d = len(pointA)
	assert(len(pointA) == len(pointB))

	distance = 0
	for i in range(d):
		temp = pointA[i] - pointB[i]
		distance += temp * temp

	return np.sqrt(distance)



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


#Script a bunch of tests w/ generated or imported matrix


d = int(5000) # Original dimensionality
n = int(2257) # Number data points
number_tests = 20 # How many tests/data points for accuracy we want to try

eps = 0.05
assert(eps > 0 and eps < 1)

#Generate a random high-dimensional matrix
#original_data = Generate_Matrix(m, N)
#categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
categories = ['sci.med']

original_data = fetch_20newsgroups(subset='train', categories=categories,
	shuffle=True, random_state=42)

word_counts = CountVectorizer().fit_transform(original_data.data)

tf_transformer = TfidfTransformer(use_idf=False).fit(word_counts)
# the results is a scipy.sparse.csr.csr_matrix
original_data = tf_transformer.transform(word_counts)

#Clip to chosen dimensions and truncate
original_data = original_data[:,:d] #words data set has 16,257 dimensions by default
original_data = normalize(original_data, 'l2')
original_data = original_data.asfptype()
#original_data = original_data.transpose() #flip from 5000x594 to 594x5000 row to col
print("Final dimensions of given data is:", original_data.shape, "n rows by d columns")

#Calculate the original distances in here
original_distances = euclidean_distances(original_data, squared=True).ravel()
nonzero_rows = original_distances != 0
original_distances = original_distances[nonzero_rows] #Ignore self distances, which are 0

k = int(Calculate_Minimum_JLT_Dimensions(n, eps) + 1) #Round up from this low value

#Space test_dimensions over x-y range w/ z attetmpts
test_dimensions = np.linspace(k, d/2, number_tests).astype(int) #
distance_errors = []

#print("----", test_dimensions, "\n", k)

for test_num, current_dim in enumerate(test_dimensions):
	print("Running test", test_num, "with dimensions", current_dim, end="")
	start_time = time.time()

	#Generate current transformation matrix
	#d rows and k columns, for transforming (d-dim, n rows) to (k-dim, n rows)
	transformation_matrix = Generate_JLT_Transform_Matrix(d, current_dim)

	print(" .", end="")

	#Transformed Matrix
	transformed = TransformMatrix(original_data, transformation_matrix)
	#JLT says to use, [15]

	print(" .", end="")

	#Test the distances found
	#outOfBounds, results = Calculate_JLT_Diste_Diffces(original_data, transformed, eps)
	transformed_distances = euclidean_distances(
		transformed, squared=True).ravel()[nonzero_rows]

	end_time = time.time()

	#See what the average distance is from the original distances
	error = transformed_distances - original_distances
	distance_errors.append(np.average(error))

	print(" . Done! %.3fs" % (end_time-start_time))

print(distance_errors)

plt.plot(test_dimensions, distance_errors, marker="*")

plt.xlabel("Number of Final Dimensions")
plt.ylabel("Average Distance Error of Transformed Data to Original Data")
plt.title("Differences in Euclidean Distances\n of Johnson-Lindenstrauss transformed Data")

print("\n All tests done! Displaying plot.\n")
plt.show()


