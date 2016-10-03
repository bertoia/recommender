import pandas as pd
import GPy
import IPython.display as display
import pylab as pb
import numpy as np

## GP model per movie

#to change filepath
movieFpath = r'C:\Users\Mitigator\Desktop\CS4246\Project\data\movie_easy.csv'
userFpath = r'C:\Users\Mitigator\Desktop\CS4246\Project\data\user_hard.csv'
ratingFpath = r'C:\Users\Mitigator\Desktop\CS4246\Project\data\rating.csv'


def getRawData(fileName):
    return pd.read_csv(fileName, encoding="ISO-8859-1")


movieRaw = getRawData(movieFpath)
userRaw = getRawData(userFpath)
ratingRaw = getRawData(ratingFpath)

# to predict the user ratings for a particular movie
# movie prediction score = real value --> Y
# @dataInputs: user demographic attributes {age, gender, } --> X


rating_user = pd.merge(ratingRaw, userRaw, how="left")
rating_user = rating_user.drop('user_id', axis=1)  # remove user_id as it's not a useful feature
rating_userResult = rating_user.ix[:, [0, 1]]  # extract out rating values as Output Y
rating_user = rating_user.drop('rating', axis=1)

# convert string into float 1 for male, -1 for female
# gender = {"M":1.0, "F":-1.0}
# rating_user['gender'] = rating_user['gender'].replace(gender)

# print (rating_user)

# For X inputs, data to be sent to GP model to iterate from 1 to N
rating_user.set_index("movie_id", inplace=True)
movie_index = 1
trainX = rating_user[['gender', 'age']].loc[movie_index]
trainx1 = trainX.iloc[:int(round(len(trainX) / 2, 0))]  # slicing, upperbound excluded
testx1 = trainX.iloc[int(round(len(trainX) / 2, 0)):]

# Y Output, data to be sent to GP model t o iterate from 1 to N
rating_userResult.set_index("movie_id", inplace=True)
movie_index = 1
trainY = rating_userResult[['rating']].loc[movie_index]

#using 50% of data as training 50% as test
trainy1 = trainY.iloc[:int(round(len(trainX) / 2, 0))]
testy1 = trainY.iloc[int(round(len(trainX) / 2, 0)):]

trainx1 = trainx1.as_matrix()
testx1 = testx1.as_matrix()
trainy1 = trainy1.as_matrix()
testy1 = testy1.as_matrix()

# print(trainX)
# print(trainY)

# print(trainx1)
# print(trainy1)

kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(trainx1, trainy1, kernel)

# returns ( mean, variance )
meanY, varY = m.predict(testx1)

act = pd.DataFrame(testy1).reset_index(drop=True)
prd = pd.DataFrame(meanY)
var = pd.DataFrame(varY)
diff = pd.concat([act, prd, var], axis=1, join_axes=[act.index])
diff.columns = ['actual', 'predicted', 'variance']
diff['residual'] = diff.apply(lambda r: r['predicted']-r['actual'], axis=1)
print(diff.sort_values(by='residual').head())


