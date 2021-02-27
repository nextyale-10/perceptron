#!/usr/bin/python3

# AUTHOR:  *Chen Yao*
# NetID:   *cyao10
# csugID:  *your csug login here (if different from NetID*

import numpy as np
from matplotlib import pyplot as plt
# TODO: understand that you should not need any other imports other than those
# already in this file; if you import something that is not installed by default
# on the csug machines, your code will crash and you will lose points


# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1]+[1] , dtype=np.float64) ## moved x_0 to the beginning. with the ordering x0 x1 x2...
    y = np.float64(tokens[-1])
    
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        #print((xs,ys))
        return xs, ys

def dot_product( v1, v2):
    dim = v1.size
    result=0
    for i in range(dim):
        
        result+=v1[i]*v2[i]
    
    return result
    
# def w_generator(dim):
    
#     list_weight=[]
#     for i in range(dim):
#         list_weight.append(np.random.rand())

#     w= np.array(list_weight,dtype=np.float64)
#     return w


def find_mis(train_data,weights,s):
    data_size= len(train_data[0])
    

    #mis_list=[]

    for i in range(s,data_size):
        test_value= train_data[1][i]*dot_product(train_data[0][i],weights)
        if test_value<=0:
            return train_data[0][i],i
        
    
    return None

    
# Do learning.
def perceptron(train_xs, train_ys, iterations):
    
    ## train_xs is a list of numpy.array ,each array is a vector
    dim= train_xs[0].size
    w= np.zeros(dim,dtype=np.float64)
    data_size=len(train_ys)
    global step
    global step_accuracy_x
    global step_accuracy_y
    total = iterations	
    global itera 
    global max_accuracy
    while True:

    	
        step_accuracy_x.append(step)
        
        acc= accuracy(w,train_xs,train_ys)

        max_accuracy=max(acc,max_accuracy)

        step_accuracy_y.append(acc)

        t=find_mis((train_xs,train_ys),w,0)
        #if step<=10:
        #   print ("{}. {}".format(step, w))
        itera =total- iterations
        
        if (itera) %1000==0 and (itera) !=0:
        
    	    print("\n{} iterations has been finished\n".format(itera))
        
       # print (iterations)
        if iterations<=0:
            print("cannot find it")
            return w
        if t!=None:
            while t!=None:
                mis_vector =t[0]
                # print("w  {}".format(w))
                # print("mis  {}".format(mis_vector))

                true_cls=train_ys[t[1]]
                #print("x={}\ny={}\n\n".format(mis_vector,true_cls))
                w= w+1*true_cls*mis_vector

                step+=1
                if step % 10000==0:
                    print("Weights have been updated {} times".format(step))
                if t[1]<data_size:    
                    t=find_mis((train_xs,train_ys),w,t[1])
                    
                
            iterations-=1
                
        elif t==None:
            print("\nfind it!")
            break

        


    



    return w


# Return the accuracy over the data using current weights.
def accuracy(weights, test_xs, test_ys):
    data_size= len(test_xs)
    
    
    sum=0
    for i in range(data_size):
        
        if dot_product(test_xs[i],weights)*test_ys[i]>0:
            sum+=1
    


    return (sum/data_size)
def max_norm(train_xs):
    max_n=0
    for v in train_xs:
        new_n= np.linalg.norm(v)
        if new_n >max_n:
            max_n=new_n


    return max_n

def main():
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    """
    train_xs, train_ys = parse_data(args.train_file)
    t0=time.process_time()
    weights = perceptron(train_xs, train_ys, args.iterations)
    t1=time.process_time()
    
    upper_R= max_norm(train_xs)


    plt.plot(step_accuracy_x,step_accuracy_y)
    plt.title("Process table")
    plt.xlabel("Times of Updates")
    plt.ylabel("Accuracy")
    print("it uses {}sec\n".format(t1-t0))
    
    a = accuracy(weights, train_xs, train_ys)
    print("All data vectors are bounded by R= {}".format(upper_R))
    print("it uses {} steps(updates) and {} iterations".format(step,itera))
    if max_accuracy ==1:
        print("Since it converges, we can calculate out delta^2 > {}".format(upper_R*upper_R/step))
    
    print('\nFinal accuracy: {}\nMax accuracy: {}'.format(a,max_accuracy))
    print('\nFeature weights (bias last): {}'.format(' '.join(map(str,weights))))
   
    plt.show()

if __name__ == '__main__':
    step =0
    itera=0
    max_accuracy=0
    step_accuracy_x =[]
    step_accuracy_y=[]
    main()
#parse_data("linearSmoke.dat")
