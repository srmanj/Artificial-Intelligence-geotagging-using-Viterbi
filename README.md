# B551 Assignment 2: Games and Bayes
##### Submission by Sri Harsha Manjunath - srmanj@iu.edu; Vijayalaxmi Bhimrao Maigur - vbmaigur@iu.edu; Disha Talreja - dtalreja@iu.edu
###### Fall 2019

## Part 2 Horizon Finding

A classic problem in computer vision is to identify where on Earth a photo was taken using visual features alone (e.g., not using GPS). For some images, this is relatively easy — a photo with the Eiffel tower in it was probably taken in Paris (or Las Vegas, or Disneyworld, or ...). But what about photos? One way of trying to geolocate such photos is by extracting the horizon (the boundary between the sky and the mountains) and using this as a “fingerprint” that can be matched with a digital elevation map to identify where the photo was taken.

Let’s consider the problem of identifying horizons in images. We’ll assume relatively clean images, where the mountain is plainly visible, there are no other objects obstructing the mountain’s ridge- line, the mountain takes up the full horizontal dimension of the image, and the sky is relatively clear. Under these assumptions, for each column of the image we need to estimate the row of the image corresponding to the boundary position. Plotting this estimated row for each column will give a horizon estimate.
We’ve given you some code that reads in an image file and produces an “edge strength map” that measures how strong the image gradient (local constrast) is at each point. We could assume that the stronger the image gradient, the higher the chance that the pixel lies along the ridgeline. So for an m × n image, this is a 2-d function that we’ll call I(x, y), measuring the strength at each pixel (x, y) ∈ [1, m] × [1, n] in the original image. Your goal is to estimate, for each column x ∈ [1, m], the row sx corresponding to the ridgeline. We can solve this problem using a Bayes net, where s1,s2,...sm correspond to the hidden variables, and the gradient data are the observed variables (specifically w1,w2,...wm, where w1 is a vector corresponding to column 1 of the gradient image).
1. Perhaps the simplest approach would be to use the Bayes Net. You’ll want to estimate the most-probable row s∗x for each column x ∈ [1, m],
s∗i =argmaxP(Si =si|w1,...,wm). si
Show the result of the detected boundary with a blue line superimposed on the image in an output file called output simple.jpg.
Hint: This is easy; if you don’t see why, try running Variable Elimination by hand on the Bayes Net.
2. A better approach would use the Bayes Net. Use the Viterbi algorithm to solve for the maximum a posterior estimate,
arg max P (S1 = s1, ..., Sm = sm|w1, ..., wm), s1 ,...,sm
and show the result of the detected boundary with a red line superimposed on the input image in an output file called output map.jpg.
3. A simple way of improving the results further would be to incorporate some human feedback in the process. Assume that a human gives you a single (x,y) point that is known to lie on the ridgeline. Modify your answer to step 2 to incorporate this additional information. Show the detected boundary in green superimposed on the input image in an output file called output human.jpg. Hint: you can do this by just tweaking the HMM’s probability distributions – no need to change the algorithm itself.
   
Run your code on our sample images (and feel free to try other sample images of your own) and include
some examples in your report.
Hints. What should the emission and transition probabilities be? There’s no right answer here, but intuitively you’ll want the emission probabilities P(wi|Si = si) to be chosen such that they are high when si is near a strong edge according to the gradient image, and low otherwise. The idea behind the transition probabilities is to encourage “smoothness” from column to column, so that the horizon line doesn’t randomly jump around. In other words, you’ll want to define P(Si+1 = si+1|Si = si) so that it’s high if si+1 ≈ si and is low if not.


#### Preprocessing of the image 

1. The image is passed through the filter that measures how strong the image gradient is at each point.

#### Approach for part 1

Finding the horizon using Bayes Net

1. Algorithm finds the max gradient value at every column. 
This approach fetches a ridgeline that is not continuous and smooth

#### Approach for part 2

Finding the horizon using viterbi Algorithm 

This algorithm involves calculating three probability:
1. Initial Probability: Initial Probability is assigned equal to all pixels in the first column i.e. reciprocal of number of rows.
Additionally, by observing we know that,  the horizon mostly is concentrated on the first 3/4th part of the image. So we increased probability of the first 3/4th part of pixel by multiplying by reciprocal of 3/4th number of pixel.

2. Transition Probability: The transition probability is reciprocal manhattan distance of the row number of pixel in col i and pixel in col i+1. Additionally, since we know that there cannot be a huge dip in the horizon, the transition probability of the pixels far away are equated to zero. 

The threshold that is considered in coding is 10. So, for the pixels whose row difference is greater than 10, the transition probability is 0. Otherwise it is reciprocal of the manhattan distance(modulus of the difference in row numbers)

3. Emission Probability: The emission probability is normalized value of the gradient i.e. ratio gradient value at that pixel to the sum of gradient value of that column.

Important Callout: Due to attenuation observed in floating point values that are used to represent probabilities, algorithm uses inverse logarithm to convert the problem into minimization problem.


#### Approach for part 3

In this approach we have an extra information that is coordinate of one pixel value that lies on the ridge line of an image.

We Consider same transition and emission probability to solve the problem. 

1. We split the image into two parts at a given column value and flip the first part of the image.
2. Assign a initial probability of 1 or 0 based of minimization or maximization problem for the row value given by the user for both the image.
i.e. the human input confirms that probability of the pixel being on the ridge line.

3. Now path for both the image is calculated using same transition, emission probability and new initial probability. These two paths are clubbed together for the result.


#### Implementation Details:

Approaches for part 2 and part 3:
1. The Probability value for every pixel is calculated and a string keeps track of the path the probability has taken to reach that pixel. 

In the last column we take the pixel with min/max probability in this case we take min probability and corresponding path.

#### Results:

![alt text](https://github.com/srmanj/Artificial-Intelligence-geotagging-using-Viterbi/blob/master/imgs/0.jpg)
![alt text](https://github.com/srmanj/Artificial-Intelligence-geotagging-using-Viterbi/blob/master/imgs/1.jpg)
![alt text](https://github.com/srmanj/Artificial-Intelligence-geotagging-using-Viterbi/blob/master/imgs/2.jpg)
![alt text](https://github.com/srmanj/Artificial-Intelligence-geotagging-using-Viterbi/blob/master/imgs/3.jpg)

**References** - </br>

[1] - https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160011500.pdf  (Idea for Viterbi implementation) </br>
[2] - https://www.cs.cornell.edu/courses/cs312/2002sp/lectures/rec21.htm . </br>




