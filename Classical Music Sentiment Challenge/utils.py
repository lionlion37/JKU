import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

########## Combine pct_10 and pct_90 #########
# Calculating their distance (pct_90 - pct_10)
# Zwar nicht mehr nützlich als Lagemaß, allerdings haben wir dafür noch Mean und Median

class PCTCombiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, inp:pd.DataFrame, y=None):

        self.pct_10, self.pct_90 = [], []
        self.pct_data = pd.DataFrame()

        for col in inp.columns:
            if 'pct_10' in col:
                self.pct_10.append(col)
            elif 'pct_90' in col:
                self.pct_90.append(col)

        pattern = '(.+)pct_10_(\d+)'
        pattern_alt = '(.+)pct'

        for p10, p90 in zip(self.pct_10, self.pct_90):
            res = re.search(pattern, p10)
            try:
                self.pct_data[res.group(1) + 'pct_dist_' + res.group(2)] = inp[p90] - inp[p10]
            except:
                res = re.search(pattern_alt, p10)
                self.pct_data[res.group(1) + 'pct_dist_10'] = inp[p90] - inp[p10]
    
    def transform(self, inp:pd.DataFrame, y=None):
        result = inp.drop(columns=self.pct_10+self.pct_90)
        result = pd.concat([result, self.pct_data], axis=1)

        return result

    def fit_transform(self, inp:pd.DataFrame, y=None):
        
        self.fit(inp)

        return self.transform(inp)

########## Mean or Median ##########
# If feature is skewed: Median, else: Mean
# skewed: |Pearson's coefficient of skewness| > 0.5

class MMCombiner(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, inp:pd.DataFrame, y=None):

        self.means, self.medians, stds = [], [], []
        self.loc_measure = pd.DataFrame()
        self.pc = []

        for col in inp.columns:
            if 'mean' in col:
                self.means.append(col)
            elif 'pct_50' in col:
                self.medians.append(col)
            elif 'std' in col:
                stds.append(col)

        self.means = self.means[6:]  # first 6 have only mean and no median
        stds = stds[6:]

        pattern = '(.+)mean_(\d+)'
        pattern_alt = '(.+)mean'

        for mean, median, std in zip(self.means, self.medians, stds):
            pcs = 3 * (inp[mean] - inp[median]) / inp[std]
            self.pc.append(pcs)
            mask = abs(pcs) > 0.5

            try:
                res = re.search(pattern, mean)
                self.loc_measure[res.group(1) + 'loc_measure_' + res.group(2)] = mask * inp[mean] + ~mask * inp[mean]
            except:
                res = re.search(pattern_alt, mean)
                self.loc_measure[res.group(1) + 'loc_measure_10'] = mask * inp[mean] + ~mask * inp[mean]

    def transform(self, inp:pd.DataFrame, y=None):

        result = inp.drop(columns=self.means+self.medians)
        result = pd.concat([result, self.loc_measure], axis=1)

        return result

    def fit_transform(self, inp:pd.DataFrame, y=None):
        
        self.fit(inp)

        return self.transform(inp)

########## Keep 10-90 Percentile distance? ##########
# drop 10-90 Pct. dist. if correlation to std. is >= 0.95

class PSCombiner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return None
    
    def fit(self, inp, y=None):

        self.correlation = inp.corr()
        self.correlation_list = []

    def transform(self, inp, threshold=0, y=None):
        result = inp.copy()
        self.threshold = threshold

        pattern = '(.+)pct_dist_(\d+)'
        pattern_alt = '(.+)pct_dist'

        for col in inp.columns:
            if 'pct' in col:
                try:
                    res = re.search(pattern, col)
                    # print(col, res.group(1) + 'std_' + res.group(2))
                    # print(self.correlation[col][res.group(1) + 'std_' + res.group(2)])

                    current_corr = self.correlation[col][res.group(1) + 'std_' + res.group(2)]
                    self.correlation_list.append(current_corr)

                    if current_corr >= self.threshold:
                        result = result.drop(col, axis=1)
                        
                except KeyError:
                    res = re.search(pattern_alt, col)
                    # print(col, res.group(1) + 'std')
                    # print(self.correlation[col][res.group(1) + 'std'])

                    current_corr = self.correlation[col][res.group(1) + 'std']
                    self.correlation_list.append(current_corr)

                    if current_corr >= self.threshold:
                        result = result.drop(col, axis=1)
        
        return result

    def fit_transform(self, inp, threshold=0, y=None):

        self.threshold = threshold
        self.fit(inp)

        return self.transform(inp)

########## Normalization and Pipeline ##########

input_pipeline = Pipeline([
    ('pct_combiner', PCTCombiner()),
    ('mm_combiner', MMCombiner()),
    ('ps_combiner', PSCombiner()),
    #('std_scaler', StandardScaler()),
])

########## Parameter Visualization ##########

def graph_overfit(X, y, X_test, y_test, model_class,default_params,param, param_vals, scale="linear"):
    results = {
    "score":[],
    "Set Type":[],
    "param":[]
    }
    for param_val in param_vals:
        model = clone(model_class)
        model.set_params(**default_params)
        model.set_params(**{param:param_val})

        model.fit(X, y)
        train_score = model.score(X,y)
        results["score"].append(train_score)
        results["Set Type"].append("Train")
        results["param"].append(param_val)

        test_score = model.score(X_test,y_test)
        results["score"].append(test_score)
        results["Set Type"].append("Test")
        results["param"].append(param_val)
    return alt.Chart(pd.DataFrame(results)).mark_line(interpolate='basis').encode(
            x=alt.X("param",scale=alt.Scale(type=scale),title=param),
            y=alt.Y("score:Q",title="Accuracy",scale=alt.Scale(zero=False),axis=alt.Axis(format='%')),
            color="Set Type:N"
        )