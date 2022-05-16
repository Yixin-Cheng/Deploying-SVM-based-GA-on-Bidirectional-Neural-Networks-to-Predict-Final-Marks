"""
THis file is for some statistical manipulations on the input patterns
"""
import pandas as pd


class Postprocessor:


    def process(self,df,path):
        """
        This function divides 4 classes into 18 classes.
        :param df: original dataframe from preprocessed dataset
        :param path: post-processed dataset path
        :return:
        """

        F = df.loc[df.iloc[:, -1] == 1].mean().round(4).to_frame().T
        P = df.loc[df.iloc[:, -1] == 2].mean().round(4).to_frame().T
        C = df.loc[df.iloc[:, -1] == 3].mean().round(4).to_frame().T
        D = df.loc[df.iloc[:, -1] == 4].mean().round(4).to_frame().T

        # Fail 3 sub-classes
        F1 = 0.9 * F - 0.1 * P
        F2 = 0.8 * F - 0.2 * P
        F3 = 0.7 * F - 0.3 * P

        # Pass 6 sub-classes
        P1 = 0.9 * P - 0.1 * F
        P2 = 0.9 * P - 0.1 * C
        P3 = 0.8 * P - 0.2 * F
        P4 = 0.8 * P - 0.2 * C
        P5 = 0.7 * P - 0.3 * F
        P6 = 0.7 * P - 0.3 * C

        # Credit 6 sub-classes

        C1 = 0.9 * C - 0.1 * P
        C2 = 0.9 * C - 0.1 * D
        C3 = 0.8 * C - 0.2 * P
        C4 = 0.8 * C - 0.2 * D
        C5 = 0.7 * C - 0.3 * P
        C6 = 0.7 * C - 0.3 * D

        # Distinction 3 sub-classes

        D1 = 0.9 * D - 0.1 * C
        D2 = 0.8 * D - 0.2 * C
        D3 = 0.7 * D - 0.3 * C

        # frame = [F, P, C, D, F1, F2, F3, P1, P2, P3, P4, P5, P6, C1, C2, C3, C4, C5, C6, D1, D2, D3]
        frame = [F, P, C, D, F1, F2, F3, P1, P2, P3, P4, P5, P6, C1, C2, C3, C4, C5, C6, D1, D2, D3]

        result = pd.concat(frame, ignore_index=True)
        # result.drop(result.columns[[1]], axis=1, inplace=True)
        # result.columns = [''] * len(result.columns)
        result.to_csv(path, header=True, index=False)
        df=pd.read_csv(path)
        return df