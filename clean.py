from sklearn.model_selection import train_test_split
import pandas as pd

def cleaned_train_and_target(df,train_feat,target_feat,inference=False):
      
    if inference == False:
        categoric_feat = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
        numeric_feat = [feat for feat in train_feat if feat not in categoric_feat]
        numeric_feat.extend(['customer_ID'])
        categoric_feat.extend(['customer_ID'])
        # aggregate rows on each customer id
        agg_numeric_X = df[numeric_feat].groupby(['customer_ID']).mean()
        agg_categoric_X = df[categoric_feat].groupby(['customer_ID']).agg(pd.Series.mode)
        agg_y = df[[target_feat+].groupby()
        X = pd.merge(agg_numeric_X,agg_categoric_X,on='customer_ID')
        y = df[target_feat]
        X = df[train_feat]
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    if inference == True:
        y_train = None
        y_test = None
        X_train = None
        X_test = df
    
    return X_train,X_test,y_train,y_test