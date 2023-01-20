import sys
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from rq3_dataset import balance
import logging as log
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import in_cols, fit_cols, enum_cols, fit_labels, CSVData

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

min_legend, mean_legend = 'RSwREP', 'RS'
fitness_labels = ['dfc', 'dfv', 'dfo', 'dt']
no_of_fitnesses = len(fitness_labels)

SEED = 0
EXP_REPEAT = 10

random_ = np.random.RandomState(seed=SEED)

def train_model(model, df, cv=5):
    y = df[df.columns.intersect(fit_cols)]
    X = df[df.columns.difference(y.columns)]
    cv = StratifiedKFold(n_splits=cv, random_state=random_)
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = str(model)
    return scores, desc

def trainDecisionTree(df, cv=5, **kwargs):
    kwargs.setdefault('max_depth', 5)
    model = DecisionTreeClassifier(**kwargs)
    return train_model(model, df, cv)

def trainSVM(df, cv=5, **kwargs):
    model = SVC(**kwargs)
    return train_model(model, df, cv)

def trainMLP(df, cv=5, **kwargs):
    kwargs.setdefault('hidden_layer_sizes', (50, 100))
    kwargs.setdefault('learning_rate', 'adaptive')
    model = MLPClassifier(**kwargs)
    return train_model(model, df, cv)

def trainModels(df, cv=5, **kwargs):
    models = []
    for i in range(4):
        df = df.drop(['F0_0', 'F0_1', 'F0_2', 'F1_0', 'F1_1', 'F1_2', 'F2_0', 'F2_1', 'F2_2', 'F3_0', 'F3_1', 'F3_2'], axis=1)
        df_train = df.sample(frac=1, random_state=random_)
        df_train = balance(df_train, output_file=None)
        scores, desc = trainDecisionTree(df_train, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores['estimator'][np.argmax(f1s)])

    return models

def balance(df, target='Y', output_file='dataset_balanced.csv'):
    targets = ['Y']
    X = df.drop(targets, axis=1).values
    y = df[target].values
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    df_res = pd.DataFrame(X_res, columns=df.drop(targets, axis=1).columns)
    df_res[target] = y_res
    if output_file is not None:
        df_res.to_csv(output_file, index=False)
    return df_res

def smartRandomSearchOneIteration(scenario, f_arr_detailed_iter, models=None, method='or'):
    scenario = list(map(int, scenario[8:].split(',')))
    dropped = [3, 4]  ##  Ego and Non-ego blueprints...
    scenario = [scenario[i] for i in range(len(scenario)) if i not in dropped]
    cnt = 0
    X = np.array([scenario])
    ys = []
    ps = []
    if models:
        ps.append(models[0].predict(X)[0] > 0.5)
    else:
        ps.append(np.random.rand() > 0.5)
    for i in range(3):  ##  max number of time steps we want to proceed...
        X = np.concatenate((X[0], f_arr_detailed_iter[i]))[np.newaxis, :]
        ys.append(f_arr_detailed_iter[i])
        if models:
            ps.append(models[i+1].predict(X)[0] > 0.5)
        else:
            ps.append(np.random.rand() > 0.5)
        cnt -=- 1
        if method == 'and' and not all(ps):
            break
        elif method == 'or' and not any(ps):
            break
    f_min = np.min(ys, axis=0)
    f_mean = np.mean(ys, axis=0)
    return f_min, f_mean, cnt

def smartRandomSearchOneRun(scenarios, f_arr_detailed, models=None, method='or'):
    f_min, f_mean, cnt = smartRandomSearchOneIteration(scenarios[0], f_arr_detailed[0], models, method)
    f_mins = [f_min]
    f_means = [f_mean]
    cnts = [cnt]
    for i in range(1, len(scenarios)):
        f_min, f_mean, cnt = smartRandomSearchOneIteration(scenarios[i], f_arr_detailed[i], models, method)
        f_mins.append([min(f_min[j], f_mins[-1][j]) for j in range(len(f_min))]) 
        f_means.append([min(f_mean[j], f_means[-1][j]) for j in range(len(f_mean))])
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)

def smartRandomSearch(scenarios, f_arr_detailed, runs=20, models=None, method='or'):
    n_iterations = len(f_arr_detailed) // runs
    f_mins = []
    f_means = []
    cnts = []
    for i in range(runs):
        f_mins_prime, f_means_prime, cnt = smartRandomSearchOneRun(scenarios[i*n_iterations:(i+1)*n_iterations], f_arr_detailed[i*n_iterations:(i+1)*n_iterations], models, method)
        f_mins.append(f_mins_prime)
        f_means.append(f_means_prime)
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)

if __name__  == '__main__':
    ##  Training models...
    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1], min_run=EXP_REPEAT)

    df_ = data.df_
    df_ = df_.groupby(level=in_cols) \
           .sample(EXP_REPEAT, random_state=random_) \
           .drop(columns=df_.columns.difference(fit_cols))
    
    df_fit = df_.copy()
    df_fit['i'] = df_fit.groupby(level=in_cols).cumcount()
    df_fit = df_fit.pivot_table(index=in_cols, columns='i', values=fit_cols).set_index(in_cols)
    df_fit.columns = list(map(lambda x: '_'.join(map(str, x)), df_fit.columns))

    df = df_[in_cols].join(df_fit) \
                     .join(df_.agg(lambda f: (f > 0).any() & (f <= 0).any())) \
                     .reset_index()
    one_hot = pd.get_dummies(df[enum_cols])
    df = df.drop(columns=enum_cols).join(one_hot)

    models = trainModels(df)
    
    ##  Smart random search random mode with or ...
    f_min_smart_random_or, f_mean_smart_random_or, cnt_random_or = smartRandomSearch(scenarios, f_arr_detailed, models=None, method='or')

    ##  Smart random search with models with or...
    f_min_smart_or, f_mean_smart_or, cnt_or = smartRandomSearch(scenarios, f_arr_detailed, models=models, method='or')

    ##  Smart random search random mode with and ...
    f_min_smart_random_and, f_mean_smart_random_and, cnt_random_and = smartRandomSearch(scenarios, f_arr_detailed, models=None, method='and')

    ##  Smart random search with models with and...
    f_min_smart_and, f_mean_smart_and, cnt_and = smartRandomSearch(scenarios, f_arr_detailed, models=models, method='and')

    ##  Random search for 10 repetitions...
    scenarios, f_arr10 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=10)
    f_min10, f_mean10 = rq1.randomSearch(f_arr10)
    min_ = np.min(f_arr10, axis=(0, 1))
    max_ = np.max(f_arr10, axis=(0, 1))
    f_min10 = (f_min10 - min_) / (max_ - min_)
    f_mean10 = (f_mean10 - min_) / (max_ - min_)
    ##  Random search for 4 repetitions...
    scenarios, f_arr4 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=4)
    f_min4, f_mean4 = rq1.randomSearch(f_arr4)
    f_min4 = (f_min4 - min_) / (max_ - min_)

    ##  Normalizing fitnesses for smart random search...
    f_min_smart_or = (f_min_smart_or - min_) / (max_ - min_)
    f_min_smart_random_or = (f_min_smart_random_or - min_) / (max_ - min_)
    f_min_smart_and = (f_min_smart_and - min_) / (max_ - min_)
    f_min_smart_random_and = (f_min_smart_random_and - min_) / (max_ - min_)


    # plotRS([f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or,f_min4, f_min10], f_mean10, ['RS-Smart-Random-AND', 'RS-Smart-models-AND', 'RS-Smart-Random-OR', 'RS-Smart-models-OR', 'RSw4REP', 'RSw10REP', 'RS'], show=True)
    plotBox([f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or,f_min4, f_min10], f_mean10, ['RS-Smart-Random-AND', 'RS-Smart-models-AND', 'RS-Smart-Random-OR', 'RS-Smart-models-OR', 'RSw4REP', 'RSw10REP', 'RS'], show=True)

    log.info(f'Number of iterations for smart RS in random mode OR: {cnt_random_or}')
    log.info(f'Number of iterations for smart RS with models OR: {cnt_or}')
    log.info(f'Number of iterations for smart RS in random mode AND: {cnt_random_and}')
    log.info(f'Number of iterations for smart RS with models AND: {cnt_and}')
