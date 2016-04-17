from random import random
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from itertools import repeat,combinations
'''
    This script shows the effect of re-scaling a variable for the training of a SVM.

    To do this
    
    1) A dataset is randomly generated and labeled (get_dataset())
    2) One of the features is re-scaled with different re-scaling factors. 
    3) For each different re-scaling, an SVM with any kernel is trained.

    The effect of the re-scaling is determined by comparing i) the classification error 
    over the training set and ii) comparing the set of the support vectors of the classifiers.

    The points and the margins are also plotted.
'''

kern = 'rbf'

def point_in_circle(x,y,r):
    return (x+(random()-0.5)*2*r, y+(random()-0.5)*2*r)


def get_dataset():
    l = []

    # Get a few separable points

    for i in range(20):
        (x,y) = point_in_circle(1,0,1.5)
        l.append((x,y,0))

        (x,y) = point_in_circle(-1,0,1.5)
        l.append((x,y,1))

    # Noisy noise

    for i in range(6): 
        (x,y) = point_in_circle(0,0,4)
        l.append((x,y,0))

        (x,y) = point_in_circle(0,0,4)
        l.append((x,y,1))
    
    # Plot the points

    with open('dataset.csv','w') as f:
        for (x,y,c) in l:
            f.write(str(x)+','+str(y)+','+str(c)+'\n')

    plt.plot([x[0] for x in l if x[2] == 0],[x[1] for x in l if x[2] ==0],'bo')
    plt.plot([x[0] for x in l if x[2] == 1],[x[1] for x in l if x[2] ==1],'ro')
    
    plt.savefig('name.png')

    return l
    


def main():
    l = get_dataset()

    y_scaling = [1,0.0000001,10000]
    print ' ----- Difference in Scores -----'
    print ''

    support_vectors = list(map(run_svm, y_scaling,repeat(l,len(y_scaling))))
    print ''
    
    print ' ----- Difference in support vectors -----'
    print ''
    labeled_sv = zip (y_scaling,support_vectors)
    
    for (svs1,svs2) in combinations(labeled_sv,2):
        difference = [x for x in svs1[1] if x not in svs2[1]]
        if len(difference)>0:
            print 'These support vectors of y_scaling =' +str(svs1[0]) + ' are not in the list of support vectors of y_scaling='+str(svs2[0])
            for x in difference:
                print ' '+ str(x)



        difference = [x for x in svs2[1] if x not in svs1[1]]
        if len(difference)>0:
            print 'These support vectors of y_scaling =' +str(svs2[0]) + ' are not in the list of support vectors of y_scaling='+str(svs1[0])
            for x in difference:
                print ' '+ str(x)

def run_svm(y_scaling,l):
    plt.clf()
    plt.plot([x[0] for x in l if x[2] == 0],[x[1]*y_scaling for x in l if x[2] ==0],'bo')
    plt.plot([x[0] for x in l if x[2] == 1],[x[1]*y_scaling for x in l if x[2] ==1],'ro')

    X = [[x[0],x[1]*y_scaling] for x in l]
    y = [x[2] for x in l]

    # Plot the points
    clf = svm.SVC(kernel = kern,C=0.1)
    clf.fit(X,y)

    s = clf.support_vectors_
    print 'Scaling: ' + str(y_scaling)

    # get the separating hyperplane
    if kern == 'linear':
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-2, 2)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        plt.plot(xx, yy, 'r-',)

        # margin

        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy + a * margin
        yy_up = yy - a * margin

        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.savefig('svn_'+str(y_scaling)+'.png')
    print 'Score = ' +str(clf.score(X,y))

    return s
main()
