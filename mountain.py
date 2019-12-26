#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np
#import collections

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

# main program
#
(input_filename, gt_row, gt_col) = sys.argv[1:]
gt_row=int(gt_row)
gt_col=int(gt_col)

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))
# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.


#Approach 1-Bayes Net 
list_bayes=[]

#Fetching 
for i in range(0,edge_strength.shape[1]):
    vec=edge_strength[0:141,i]
    vec=list(vec)
    val=max(vec[0:int(edge_strength.shape[1])])
    ind=np.where(vec==val)[0][0]
    list_bayes.append(ind)

# output answer
imageio.imwrite("output_simple.jpg", draw_edge(input_image, list_bayes, (0, 0, 255), 5))

#Approach 2- Viterbi Algorithm

# Transition Probability
######################################################################################################################################################
def transition_probability(edge_strength,row,col,Prob_hmm):
    possible_prob=[]
    for dummy in range(0,edge_strength.shape[0]):
            if row!=dummy and abs(dummy-row)<10:
                trans_prob=-np.log((1/abs(dummy-row)))
            elif row!=dummy and abs(dummy-row)>10:
                trans_prob=-np.log(0.1)
            else:
                trans_prob=-np.log(0.1)
            
            possible_prob.append(Prob_hmm[(dummy,col-1)][0]+trans_prob)
    
    return possible_prob

#Emission Probability
def emission_probability(edge_strength,row,col):
    prob_=1/int(edge_strength.shape[0]-(edge_strength.shape[0]/1.7))
    #prob_=1
    if row<int(edge_strength.shape[0]/1.7):
        return -np.log((edge_strength[row,col]/sum(edge_strength[:,col])))
    else:
        return -np.log((edge_strength[row,col]/sum(edge_strength[:,col]))*prob_)
	

def pixel_probability(edge_strength,w):
	# assigning a dictionary to hold probability 

	#Prob_hmm = collections.defaultdict(lambda : (0,""))
	Prob_hmm={}
	# Finding the initial probability
	for row in range(0,edge_strength.shape[0]):
		Prob_hmm[(row,0)]=(w[row]*emission_probability(edge_strength,row,0),"")


	# Finding the other probability
	for col in range(1,edge_strength.shape[1]):
		for row in range(0,edge_strength.shape[0]):
			possible_prob=transition_probability(edge_strength,row,col,Prob_hmm)
			emission_prob=emission_probability(edge_strength,row,col)
			prob_val=np.min(possible_prob)+emission_prob
			path=Prob_hmm[(possible_prob.index(min(possible_prob)),col-1)][1]+" "+str(possible_prob.index(min(possible_prob)))
			Prob_hmm[(row,col)]=(prob_val,path)

	#getting last column

	last_col=[]
	for i in range(0,edge_strength.shape[0]):
		last_col.append(Prob_hmm[(i,edge_strength.shape[1]-1)][0])


	#getting the taken by the max probability
	final_path_index=last_col.index(max(last_col))

	final_path=Prob_hmm[(final_path_index,edge_strength.shape[1]-1)][1].split(" ")

	final_path=final_path[1:]

	final_path=[int(i) for i in final_path]
	return final_path

######################################################################################################################################################

#initial probability
w_part2=np.full(edge_strength.shape[0],-np.log(1/edge_strength.shape[0]))
final_path=pixel_probability(edge_strength,w_part2)

#output
imageio.imwrite("output_map.jpg", draw_edge(input_image, final_path, (255, 0, 0), 5))

######################################################################################################################################################
#Approach 3 - Human Input Approach
edge_strength_1= edge_strength[:,0:gt_col]
edge_strength_2=edge_strength[:,gt_col:edge_strength.shape[1]]
#handling the column error
if gt_col!=0:
    edge_strength_1=np.flip(edge_strength_1, 1)


#initial probability
w_part3=np.ones(edge_strength.shape[0])
w_part3[gt_row]=0.01

if gt_col!=0:
    final_path_part1=pixel_probability(edge_strength_1,w_part3)[::-1]
final_path_part2=pixel_probability(edge_strength_2,w_part3)
final_path_a3=final_path_part1[:-1]+final_path_part2

imageio.imwrite("output_human.jpg", draw_edge(input_image, final_path_a3, (0, 255, 0), 5))
