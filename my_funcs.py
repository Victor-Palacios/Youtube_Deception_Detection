import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib import rcParams

from math import floor, ceil
import pandas as pd
import numpy as np
import itertools

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# For Thresholding
import scipy

# Breusch-Pagan for Heteroskedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

# Robust Regression
from sklearn.linear_model import HuberRegressor

# Validating Models
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# SMF Linear Regression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# sklearn Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# # Functions

# In[2]:


def round_down(num, divisor):
    return floor(num / divisor) * divisor

def round_up(num, divisor):
    return ceil(num / divisor) * divisor


# In[3]:


def vheatmap(df, height=7, width=7):
    corr = df.corr()
    corr = np.abs(corr)

    for i in range(len(corr)):
        for j in range(len(corr)):
            if i<=j:
                corr.iloc[i,j] = 0

    labels = []
    for string in corr.index:
        if "_"  in string:
            new_string = string[:4] + '\n' + string[string.find('_')+1:string.find('_')+5]
        else:
            new_string = string[:4]

        labels.append(new_string)

    total_columns = len(corr.index)

    fig, ax = plt.subplots(figsize=(width, height))

    ax.imshow(corr, cmap='Blues', vmin=0, vmax=1.0)

    #Add correlation to each box
    for i in range(total_columns):
        for j in range(total_columns):
            if i>j:
                color = "white"
                if corr.iloc[i,j] < .5:
                    color = "black"
                ax.text(j, i, 
                        f"{corr.iloc[i,j]:.2f}", 
                        color=color, 
                        size=16, 
                        horizontalalignment='center',
                        verticalalignment="center"
                       )          
    # spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    # x and y axis labels
    maximum = total_columns
    # ax.set_xlim(0, maximum)
    ax.set_xticks(np.arange(0, maximum))
    ax.set_xticklabels(labels)

    # ax.set_ylim(0, maximum)
    ax.set_yticks(np.arange(0,maximum))
    ax.set_yticklabels(labels)

    ax.tick_params(labelsize=18)
    ax.set_title('Correlation Heat Map', loc='left', fontdict={'fontsize': 24})
    return plt.show()


# In[4]:


def external_student_resid(model, df, influence):
    # threshold externally studentized residuals
    n = len(df)
    p = len(model.params) - 1
    
    # alpha = 0.05
    seuil_stud = scipy.stats.t.ppf(0.975, df=n-p-1)

    # detection - absolute value > threshold
    reg_studs = influence.resid_studentized_external
    atyp_stud = np.abs(reg_studs) > seuil_stud

    return list(zip(df.index[atyp_stud], list(map(lambda x: round(x,3), reg_studs[atyp_stud]))))


# In[5]:


def highest_leverage_point(influence):
    leverage = influence.hat_matrix_diag
    return (list(leverage).index(max(leverage)), (max(leverage)))


# In[6]:


def abnormal_cooks_distance(df, influence):
    n = len(df)
    
    inflsum = influence.summary_frame()
    
    reg_cook = inflsum.cooks_d

    atyp_cook = np.abs(reg_cook) >= 4/n
    
    return list(zip(df.index[atyp_cook], list(map(lambda x: round(x,3), reg_cook[atyp_cook]))))


# In[7]:


def heteroscedasticity(model):
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print("Statistic: ", round(bp_test[0],4), "P-value", round(bp_test[1],4))


# In[8]:


def resid_vs_fitted(model):
    residuals = model.resid
    fitvals = model.fittedvalues
    plt.scatter(fitvals,residuals)
    plt.xlabel("Fitted Values (Predictions)")
    plt.ylabel("Residuals")
    plt.title("Fitted Values vs. Residuals")
    plt.show()


# In[9]:


def vif(model_string, df):
    _, X = dmatrices(model_string, data=df, return_type='dataframe')

    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    return vif


# In[10]:


def make_sm_string(df, target):
    predictors = list(df.drop([target], axis=1).columns)

    model_string = target + " ~ "
    for index, p in enumerate(predictors):
        if predictors[-1] != predictors[index]:
            model_string += p + " + "
        else:
            model_string += p
    return model_string


# In[11]:


def rename_columns(df):
    dic = dict()

    for column in df.columns:
        dic[column] = column.replace(" ","_")

    return df.rename(dic, axis=1)


# In[12]:


def column_details(df, column_index):
    print(df.iloc[:,column_index].value_counts(dropna=False)[:5])
    print()
    print("number of categories:", len(df.iloc[:,column_index].value_counts(dropna=False)))
    print()


# In[13]:


def identify_iflu_pts(df, model):
    infl = model.get_influence()

    studs = external_student_resid(model=model, df=df, influence=infl)

    cookers = abnormal_cooks_distance(df=df, influence=infl)
    
    studs_set = set()
    for element in studs:
        studs_set.add(element[0])
    
    cookers_set = set()
    for element in cookers:
        cookers_set.add(element[0])
    
    influential_points = studs_set.intersection(cookers_set)
    
    return list(influential_points)


# In[14]:


def multi_scatterplot(df, target, height=10, width=20):
    plt.figure(figsize=(width, height))
    no_target_df = df.drop(target, axis=1)

    # i: index
    for i, col in enumerate(no_target_df.columns):
        plt.subplot(4, 3, i+1)
        x = no_target_df[col]
        y = df[target]
        plt.plot(x, y, 'o')

        plt.title(col)
        plt.ylabel(target)
        plt.xlabel(col)
        
        plt.tight_layout()


# In[15]:


def multi_heteroscedasticity(df, model, target, height=10, width=20):
    plt.figure(figsize=(width, height))
    no_target_df = df.drop(target, axis=1)

    # i: index
    for i, col in enumerate(no_target_df.columns):
        plt.subplot(4, 3, i+1)
        x = no_target_df[col]
        y = model.resid
        plt.plot(x, y, 'o')

        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('residuals')
        plt.tight_layout()


# In[16]:


def mallows_c(full_model, reduced_model):
    # number of parameters in the reduced model
    k = len(reduced_model.params)
    
    # number of data points
    n = len(reduced_model.fittedvalues)
    
    sse = reduced_model.mse_resid*(n - k)

    cp = (sse/full_model.mse_resid) - (n - 2*k)

    return cp


# In[17]:


def make_sm_string(df, target):
    predictors = list(df.drop([target], axis=1).columns)

    model_string = target + " ~ "
    for index, p in enumerate(predictors):
        if predictors[-1] != predictors[index]:
            model_string += p + " + "
        else:
            model_string += p
    return model_string


# In[18]:


def combo_string(combo, target):
    predictors = list(combo)

    model_string = target + " ~ "
    for index, p in enumerate(predictors):
        if predictors[-1] != predictors[index]:
            model_string += p + " + "
        else:
            model_string += p
    return model_string


# In[19]:


def best_subset(df, y):
    Y = df[y]
    X = df.drop(columns=y, axis=1)
    k = len(X.columns)

    smf_full_string = make_sm_string(df, y)
    full_model = smf.ols(smf_full_string, data=df).fit()

    num_features, feature_list, Adj_Rsq, Mallows_C, AIC, BIC  = [], [], [], [], [], []

    # Looping over k = min to k = max features in X
    for k in range(1,len(X.columns) + 1):

        # Looping over all possible combinations: from max choose k
        for combo in itertools.combinations(X.columns, k):
            smf_string = combo_string(combo, y) 
            tmp_model = smf.ols(smf_string, data=df).fit() 
            #Rsquared.append((round(tmp_model.rsquared,3)))
            Adj_Rsq.append((round(tmp_model.rsquared_adj,3)))
            Mallows_C.append((round(mallows_c(full_model, tmp_model),3)))
            feature_list.append(combo)
            num_features.append(len(combo))   
            AIC.append((round(tmp_model.aic,3)))
            BIC.append((round(tmp_model.bic,3)))

    # Store in DataFrame
    df = pd.DataFrame({'numb_features': num_features,'Mallows_C': Mallows_C, 'Adj_R^2': Adj_Rsq, 'AIC': AIC, 'BIC': BIC, 'features': feature_list})
    
    return df

# In[ ]:




