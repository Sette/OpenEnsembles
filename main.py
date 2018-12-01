import pandas as pd
from sklearn import datasets
import openensembles as oe
import matplotlib.pyplot as plt
import seaborn as sns
#Set up a dataset and put in pandas DataFrame.
x, y = datasets.make_moons(n_samples=200, shuffle=True, noise=0.02, random_state=None)
df = pd.DataFrame(x)
#instantiate the oe data object
dataObj = oe.data(df, [1,2])
#instantiate an oe clustering object
c = oe.cluster(dataObj)
c_MV_arr = []
val_arr = []
for i in range(0,39):
    # add a new clustering solution, with a unique name
    name = 'kmeans_' + str(i)
    c.cluster('parent', 'kmeans', name, K=16, init = 'random', n_init = 1)
    # calculate a new majority vote solution, where c has one more solution on each iteration
    c_MV_arr.append(c.finish_majority_vote(threshold=0.5))
    #calculate the determinant ratio metric for each majority vote solution
    v = oe.validation(dataObj, c_MV_arr[i])
    val_name = v.calculate('det_ratio', 'majority_vote', 'parent')
    val_arr.append(v.validation[val_name])

#calculate the co-occurrence matrix
coMat = c.co_occurrence_matrix()
coMat.plot(labels=False)