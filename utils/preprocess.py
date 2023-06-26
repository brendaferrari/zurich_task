import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA

class Preprocess:

    def __init__(self, data):
        self.data = data

    def rm_NaN(self):

        shape_before = self.data.shape
        print(f'Before removing NaN values: {shape_before}\n')

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        header = list(self.data.columns.values)
        A = self.data.isnull().any()
        for fact in enumerate(A) :
            if fact[1] == True :
                print("NaN values found.")
                col_Nan = header[fact[0]]
                del self.data[col_Nan]

        shape_after = self.data.shape
        if shape_after == shape_before:
            print("NaN values not found.")
        print(f'After removing NaN values: {shape_after}')

        return self.data

    def rm_EmptyFeature(self) :

        shape_before = self.data.shape
        print(f'Before removing Empty Feature: {shape_before}\n')

        zero_desc = []
        header = list(self.data.columns.values)
        for i in header :
            flag = 0
            column = self.data[i]
            column = list(column)
            stan = column[0]
            for d in column :
                if d != stan :
                    flag = 1
                    break
            if flag == 0 :
                zero_desc.append(i)
        self.data = self.data.drop(zero_desc, axis=1)

        shape_after = self.data.shape
        if shape_after == shape_before:
            print("Empty Feature values not found.")
        print(f'After removing Empty Feature: {shape_after}')
        
        return self.data

    def rm_HighCol(self, thresh):
        """
        Remove Columns with Colinearity Higher than a given threshold using a correlation matrix method
        collinearity, in statistics, correlation between predictor variables (or independent variables), such that they express a 
        linear relationship in a regression model. 
        When predictor variables in the same regression model are correlated, they cannot independently predict the value of the dependent variable.
        Multicollinearity makes it hard to interpret your coefficients, and it reduces the power of your model to identify independent variables that are statistically significant
        strong correlation exists between them, making it difficult or impossible to estimate their individual regression coefficients reliably
        """
        shape_before = self.data.shape
        print(f'Before removing High Colinearity values: {shape_before}\n')

        # Build a pearson correlation matrix
        cor_matrix = self.data.corr(method ='pearson').abs()
                    
        # Convert the matrix in a upper triangular Matrix given the colinearity simmetry
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
                    
        # Write a list of columns with high colinearity to be dropped from original self.dataset
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > thresh)]
                    
        # Drop Selected columns and store self.dataframe
        self.data = self.data.drop(to_drop, axis=1)
        shape_after = self.data.shape
        if shape_after == shape_before:
            print("High Colinearity value not found.")
        print(f'After removing High Colinearity values: {shape_after}')               
        return self.data

    def do_DataScaling(self, scaler):
        
        header = list(self.data.columns.values)
        if scaler == 'normalization':
            scaler_ = MinMaxScaler()
        if scaler == 'standardization':
            scaler_ = StandardScaler()
        data_scaled = scaler_.fit_transform(self.data)
        self.data = pd.DataFrame(data_scaled, columns=header)

        return self.data

    def get_LowVarianceColumns(self, thresh=0.0,
                                autoremove=False):
        """
        Wrapper for sklearn VarianceThreshold for use on pandas self.dataframes.
        """
        print("Finding low-variance features.")
        try:

            allColumns = self.data.columns

            X = self.data.values

            vt = VarianceThreshold(threshold=thresh)
            vt.fit(X)

            feature_indices = vt.get_support(indices=True)
            feature_names = [allColumns[idx]
                                    for idx, _
                                    in enumerate(allColumns)
                                    if idx
                                    in feature_indices]

            # get the columns to be removed
            removed_features = list(np.setdiff1d(allColumns,
                                                        feature_names))
            print("Found {0} low-variance columns."
                        .format(len(removed_features)))

            if autoremove:
                print("Removing low-variance features.")
                # remove the low-variance columns
                X_removed = vt.transform(X)

                print("Reassembling the self.dataframe (with low-variance "
                            "features removed).")
                # re-assemble the self.dataframe
                self.data = pd.DataFrame(data=X_removed, columns=feature_names)
                print("Succesfully removed low-variance columns.")

                    # do not remove columns
            else:
                print("No changes have been made to the self.dataframe.")
        
        except Exception as e:
            print(e)
            print("Could not remove low-variance features. Something "
                "went wrong.")
            pass

        return self.data

    def outlier_removal_IQR(self):
        """
        Remove outliers using the inter quartile range technique
        """
        Q1 = self.data['target'].quantile(0.25)
        Q3 = self.data['target'].quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range. 

        filter_ = (self.data['target'] >= Q1 - 1.5 * IQR) & (self.data['target'] <= Q3 + 1.5 *IQR)
        self.data = self.data.loc[filter_]  

        print(self.data.shape)
        
        return self.data
    
    def outlier_treatment(self, method='winsorize'):
            """
            Remove outliers using the inter quartile range technique
            https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-3-dcb54abaf7b0#:~:text=robust)%20Mahalanobis%20distance.-,Imputation,based%20on%20the%20remaining%20data.
            https://ndgigliotti.medium.com/trimming-vs-winsorizing-outliers-e5cae0bf22cb
            """
            Q1 = self.data['target'].quantile(0.25)
            Q3 = self.data['target'].quantile(0.75)
            IQR = Q3 - Q1    #IQR is interquartile range. 

            inner_fence_ub = Q1 - 1.5 * IQR
            inner_fence_lb = Q3 + 1.5 *IQR

            outer_fence_ub = Q1 - 3 * IQR
            outer_fence_lb = Q3 + 3 *IQR

            if method == 'winsorize':
                    print(outer_fence_lb)
                    print(outer_fence_ub)
                    print('86% quantile:   ', self.data['target'].quantile(0.86))       #10.75
                    print('89.5% quantile:   ', self.data['target'].quantile(0.895))       #10.75
                    print('90% quantile:   ', self.data['target'].quantile(0.90))       #10.75
                    print('92.5% quantile: ', self.data['target'].quantile(0.925))      #13.54
                    print('95% quantile:   ', self.data['target'].quantile(0.95))       #15.79
                    print('97.5% quantile: ', self.data['target'].quantile(0.975))      #23.93
                    print('99% quantile:   ', self.data['target'].quantile(0.99))       #41.37
                    print('99.9% quantile: ', self.data['target'].quantile(0.999))      #81.18

                    data_w = self.data.copy(deep=True)

                    #Winsorize on right-tail
                    data_w['target_wins_89.5%'] = winsorize(self.data['target'], limits=(0, 0.105))
                    data_w['target_wins_86%'] = winsorize(self.data['target'], limits=(0, 0.14))
            
                    return data_w

            if method == 'imputation':   
                    outliers_prob = []
                    outliers_poss = []
                    for index, x in enumerate(self.data['target']):
                            if x <= outer_fence_ub or x >= outer_fence_lb:
                                    outliers_prob.append(index)
                    for index, x in enumerate(self.data['target']):
                            if x <= inner_fence_ub or x >= inner_fence_lb:
                                    outliers_poss.append(index)

                    for i in outliers_prob:
                            self.data.at[i, 'target'] = None

                    #Define imputer
                    #imputer = IterativeImputer(estimator=LinearRegression(),
                                            # n_nearest_features=None,
                                            # imputation_order='ascending')
                                            # #, sample_posterior=True)
                    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                    #Fit imputer and transform                          
                    imputer.fit(self.data)
                    df_imp_tf = imputer.transform(self.data)
                    df_imp = pd.DataFrame(df_imp_tf, columns = self.data.columns)
                    
                    return df_imp
            
    def do_DimensionalityReduction(self, features):

        pca = PCA(n_components=features)
        # X is the matrix transposed (n samples on the rows, m features on the columns)
        pca.fit(self.data)

        data_x_new = pd.DataFrame(pca.transform(self.data))
        
        return data_x_new