from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def cleaned_train_and_target(df,target_feat,inference=False):
      
    if inference == False:
        categoric_feat = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_68']
        numeric_feat = [feat for feat in df.columns if feat not in categoric_feat and feat not in ['customer_ID','target']]
        numeric_feat.extend(['customer_ID'])
        categoric_feat.extend(['customer_ID'])
        # aggregate rows on each customer id
        agg_numeric_df = df[numeric_feat].groupby(['customer_ID']).mean()
        # agg_categoric_df = df[categoric_feat].groupby(['customer_ID']).agg(pd.Series.mode)
        agg_target_df = df[[target_feat,'customer_ID']].groupby(['customer_ID']).mean()
        # train_data = pd.merge(pd.merge(agg_numeric_df,agg_categoric_df,on='customer_ID'),agg_target_df,on='customer_ID')
        train_data = pd.merge(agg_numeric_df,agg_target_df,on='customer_ID')
        y = train_data[target_feat]
        cols = [col for col in train_data.columns if col not in ['target','S_2'] and col not in categoric_feat]
        X = train_data[cols]
        # X.replace({pd.NA: np.nan})
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    if inference == True:
        y_train = None
        y_test = None
        X_train = None
        X_test = df
    
    return X_train,X_test,y_train,y_test