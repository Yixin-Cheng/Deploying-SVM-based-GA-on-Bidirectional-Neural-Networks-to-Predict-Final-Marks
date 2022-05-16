"""This file is all about creating new dataset and implementing pre-processing"""
import pandas as pd
import re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

class Preprocessor:

    def __init__(self, path,new_path):
        self.path = path
        self.new_path = new_path
        self.organise()
        self.df = pd.read_csv(self.new_path)
        self.transformation()
        self.impute_missing_value()

    # 1. create a new file and organise column and rows
    def organise(self):
        """
        this function is for creating new dataset and organising the column and their corresponding values
        :param initial_path:
        :param path: old dataset path
        :return: new data path
       """
        self.df = pd.read_csv(self.path)
        self.new = []
        for i in range(5, len(self.df)):
            temp = self.df.iloc[i].values[0]
            self.new.append(temp.split())
        new_dataset = pd.DataFrame(self.new,
                                   columns=['Regno', 'Crse/Prog', 'S', 'ES', 'Tutgroup', 'lab2', 'tutass', 'lab4', 'h1',
                                            'h2', 'lab7', 'p1', 'f1', 'mid', 'lab10', 'final'])
        new_dataset.to_csv(self.new_path, index=False)


    def transformation(self):
        '''
        step 1:transformation
        step 2: outlier detection
        :return:
        '''
        # 1. for Turtgroup, scale from 1 to 10 represents 1-10 tutorials then normalize them into 0-1.
        df = pd.read_csv(self.new_path)
        for i in range(len(df['Tutgroup'])):
            if df['Tutgroup'][i] != '?':
                df.loc[i, 'Tutgroup'] = float(re.findall(r"\d+\.?\d*", df['Tutgroup'][i])[0])/10
            if df['Tutgroup'][i] == '?':
                df.loc[i, 'Tutgroup'] = np.nan

        # 2. for Cre/Prog, get the course number then normalize them into 0-1
        for i in range(len(df['Crse/Prog'])):
            df.loc[i, 'Crse/Prog'] = float(df['Crse/Prog'][i][:4])/10000
        # 3. for S, convert from int to float
        for i in range(len(df['S'])):
            df.loc[i, 'S'] = round(float(df['S'][i])/3,4)

        # 4. for ES, transform it to number from 1-3 respectively represnting F, FS and FL then normalize them into 0-1
        for i in range(len(df['ES'])):
            if df['ES'][i] == 'F':
                df.loc[i, 'ES'] = 1
            if df['ES'][i] == 'FS':
                df.loc[i, 'ES'] = 2
            if df['ES'][i] == 'FL':
                df.loc[i, 'ES'] = 3

        # 5. replace ? . to nan
        df.replace('.', np.nan, inplace=True)



        # 6. for grades, scale from 0-1 represents from 0%-100% respectively with using min-max normalisation since this method
        # would not produce negative values.

        for i in range(len(df['lab2'])):
            if df['lab2'][i] != None:
                df.loc[i, 'lab2'] = round(float(df['lab2'][i]) / 3, 4)
        for i in range(len(df['tutass'])):
            if df['tutass'][i] != None:
                df.loc[i, 'tutass'] = round(float(df['tutass'][i]) / 5, 4)
        for i in range(len(df['lab4'])):
            if df['lab4'][i] != None:
                df.loc[i, 'lab4'] = round(float(df['lab4'][i]) / 3, 4)
        for i in range(len(df['h1'])):
            if df['h1'][i] != None:
                df.loc[i, 'h1'] = round(float(df['h1'][i]) / 20, 4)
        for i in range(len(df['h2'])):
            if df['h2'][i] != None:
                df.loc[i, 'h2'] = round(float(df['h2'][i]) / 20, 4)
        for i in range(len(df['lab7'])):
            if df['lab7'][i] != None:
                df.loc[i, 'lab7'] = round(float(df['lab7'][i]) / 3, 4)
        for i in range(len(df['p1'])):
            if df['p1'][i] != None:
                df.loc[i, 'p1'] = round((float(df['p1'][i]) / 20), 4)
        for i in range(len(df['f1'])):
            if df['f1'][i] != None:
                df.loc[i, 'f1'] = round(float(df['f1'][i]) / 20, 4)
        for i in range(len(df['mid'])):
            if df['mid'][i] != None:
                df.loc[i, 'mid'] = round(float(df['mid'][i]) / 45, 4)
        for i in range(len(df['lab10'])):
            if df['lab10'][i] != None:
                df.loc[i, 'lab10'] = round(float(df['lab10'][i]) / 3, 4)

        # 7. remove the first column
        del df['Regno']

        # 8. normalize the final column to 1 to 4 to represnt F to D.
        for i in range(len(df['final'])):
            if float(df['final'][i]) >= 75:
                df.loc[i, 'final'] = 1
            elif float(df['final'][i]) >= 65 and float(df['final'][i]) <= 74:
                df.loc[i, 'final'] = 2
            elif float(df['final'][i]) >= 50 and float(df['final'][i]) <= 64:
                df.loc[i, 'final'] = 3
            elif float(df['final'][i]) < 50:
                df.loc[i, 'final'] = 4
        df.to_csv(self.new_path, index=False)


    def impute_missing_value(self):
        """
        this function written for handling missing values wtih using random forest
        the order of imputation is from the least column to the most ones first, fill the missing values with 0 in other columns
        then run the algorithm and do the iteration.
        """
        df = pd.read_csv(self.new_path)
        y_full = pd.read_csv(self.new_path)
        X_missing_reg = df.copy()  # temp df
        sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
        sortindex = list(sortindex)
        sortindex.remove(0)  # remove the columns that are without missing values
        sortindex.remove(1)
        sortindex.remove(2)
        for i in sortindex:
            df = X_missing_reg
            # construct new pattern matrix
            fillc = df.iloc[:, i]  # all rows
            df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
            # fill 0 in all cells that have missing values
            df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]
            # use random forest to impute missing value
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(Xtrain, Ytrain)
            Ypredict = rfc.predict(Xtest)
            Ypredict = [round(element, 4) for element in Ypredict]
            X_missing_reg.loc[df.iloc[:, i].isnull(), df.columns[i]] = Ypredict  # fill new missing values in the cells

        X_missing_reg.to_csv(self.new_path, header=False, index=True) # remove header in the end