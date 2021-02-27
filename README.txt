

1.Definition for the terms:
	
  1)Iteration: This iteration means 1 traverse from beginning of the data set to the end. For example 100 iterations means the program has traversed 100 time   all data points.
  
  2)Step: Step means the number of times of updates for weight vector. In 1     iteration, steps can be increased by multiple times.(My report graph is based on steps instead of iterations.)

  3)Accuracy: Accuracy means the rate of correct-classified vectors over total  number of vectors. 
  
  4)R: Here, R, without losing generality, is defined as the maximum norm for
all data  vectors.

 
2.Description for the project:
  
  This project is an implementation for perceptron algorithm. All functions are contained in the file named "perceptron_starter.py" The main function is named  "perceptron". The file also contains several helper function such as dot_product and find_misclassified.

3.How to run the program:
  "perceptron_starter"takes 2 arguments from command-line. 1.train_file and 
 2.iterations. 
for train_file, we only need to enter a string with the following format:
data/{name of data file}
for iterations, we need to enter a int.

For example :

python3 perceptron_starter.py --train_file data/challenge4.dat --iterations 250000

This command will run "perceptron_starter.py" with "challenge4.dat" and iteratesall data for 250K times.
