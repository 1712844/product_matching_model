import pandas as pd
import re
import numpy as np
import featureExtractors as fe
import myFeatureExtractors as mfe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
import multiprocessing as mp
import pickle
from fuzzywuzzy import fuzz
from rapidfuzz import process

#Initial training process
#df= pd.read_excel('../finals.xlsx', sheet_name='Sheet1')
#df= pd.read_excel('../finals.xlsx', sheet_name='Sheet1')
pkl_filename = "fuzzy_MLP_v2.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

def transform(X):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X)
    return X_train_std
#pkl_filename = "pickle__model.pkl"
df= pd.read_excel('getir.xlsx', sheet_name='getir')

#with open(pkl_filename, 'rb') as file:
#    pickle_model = pickle.load(file)

def fuzzy_filter(row_in, df_out):
    df_name = df_out['name']
    df_fuzz = pd.DataFrame(process.extract(row_in, df_name, scorer = fuzz.token_sort_ratio, limit = 10, score_cutoff = 80), columns=("matcheString", "matchScore", "index"))
    return df_fuzz

def process_dataframe(df):
   df['uniqueNumberCount'] = df.apply(fe.get_unique_number_count, axis=1)
   df['numberMatchRate'] = df.apply(fe.get_rate, axis=1)
   #df['matchScore'] = df.apply(fe.fuzzy_rate, axis=1)
   df['normalizedMatchRate'] = (df['numberMatchRate']+2).apply(np.log)
   df['squaredPriceRate'] = df['ti_le_gia']**2
   return df
#--------------------------------------------------------

#--------------------------------------------------------

df_in = pd.read_excel('DNS_2/Phu kien may tinh - input - 200.xlsx', sheet_name='Sheet1')
#print(df_in)
#print(df_in.shape[0])
df_out = pd.read_excel('DNS_2/Phu kien may tinh - output - 200.xlsx', sheet_name='Sheet1')
#print(df_out.shape[0])
#print(list(df_out.columns))
#print(df_out)
class OneToMany:
    def __init__(self, row, df):
        self.df = df
        self.row = row
    def createOneToManyDataframe(self):
        result = pd.DataFrame(columns=('id','id_match', 'product1', 'product2', 'gia', 'gia2'))

        df_fuzzy_filter = fuzzy_filter(self.row['name'], self.df)
        
        if(df_fuzzy_filter.empty):
            return result
        else:
            sub_index = df_fuzzy_filter['index'].values
            df_filtered = self.df.iloc[sub_index]
        # filtered output
        #
            result[['id_match', 'product2', 'gia2',]] = df_filtered[['id', 'name', 'price']]
            result[['id', 'product1', 'gia']] = [self.row['id'], self.row['name'], self.row['price']]
            result['matchScore'] = df_fuzzy_filter['matchScore'].div(100).values
            result['ti_le_gia'] = result['gia']/result['gia2']
        return result

def unchange_row(row_in):
    final_temp = {
        'id': row_in.id,
        'id_match': row_in.id,
        'name': row_in['name'],
        'price': row_in.price,
    }
    return final_temp

class Predictions:
    def __init__(self, df_in, df_out, df_train):
        self.df_in = df_in
        self.df_out = df_out
        self.df_train = df_train
    def predict_on_dataframes(self):
        #new_product_id = 0
        #final_results = pd.DataFrame(columns=('id','id_match','product1, 'product_out','gia', 'mota'))
        for index, row_in in self.df_in.iterrows():
            #Create a dataframe with the input product and output products
            otm = OneToMany(row_in, self.df_out)
            process_otm = otm.createOneToManyDataframe()
            if(process_otm.empty):
                final_temp = unchange_row(row_in)
                self.df_out = self.df_out.append(final_temp, ignore_index = True)
            else:
                rd = process_dataframe(process_otm)[['matchScore','squaredPriceRate', 'uniqueNumberCount', 'normalizedMatchRate']].values
                rd = transform(rd)
                process_otm['match'] = pickle_model.predict(rd)

                condition = (process_otm.match == 1) & (process_otm.numberMatchRate >= 0.65)
                p = process_otm[condition] 
                #print(p)
                #print(process_otm.empty)
                #print(p.empty)
                if(p.empty):
                    #print(process_otm[['match', 'matchScore', 'squaredPriceRate', 'numberMatchRate']]) 
                    final_temp = unchange_row(row_in)
                    print('not matched')
                    #print(final_temp)
                    self.df_out = self.df_out.append(final_temp, ignore_index = True)
                else:
                    #print(rd)
                    pos = p['squaredPriceRate'].sub(1).abs().idxmin()
                
                    #print(pos)
                    result = process_otm.loc[pos]
                    #print('squaredPriceRate: ' + str(result['squaredPriceRate']))
                    #print('matchScore: ' + str(result['matchScore']))
                    #print('uniqueNumberCount: ' + str(result['uniqueNumberCount'])) 
                    #print('numberMatchRate: ' + str(result['numberMatchRate']))

                    final_temp = {
                            'id': result.id,
                            'id_match': result.id_match,
                            'name': result.product1,
                            'price': result.gia,
                        }

                    train_temp = {
                            'product1': result.product1,
                            'product2': result.product2,
                            'gia': result.gia,
                            'gia2': result.gia2,
                            'ti_le_gia': result.ti_le_gia,
                            'squaredPrieRate': result.squaredPriceRate,
                            'matchScore': result.matchScore,
                            'uniqueNumberCount': result.uniqueNumberCount,
                            'numberMatchRate': result.numberMatchRate,
                            'Match': 1
                        }
                        #print(final_temp)
                    print(train_temp)
                        #print(final_temp['ten'] + ' ---- ' + final_temp['product_out'])                   
                    self.df_out = self.df_out.append(final_temp, ignore_index = True)
                    self.df_train = self.df_train.append(train_temp, ignore_index = True)            
        return self.df_out, self.df_train


df_train = pd.DataFrame()

a = Predictions(df_in, df_out, df_train)
#a.predict_on_dataframes()
finals_out, finals_train = a.predict_on_dataframes()
print("done")
print(finals_train)
finals_out.to_excel(r'DNS_2/Phu kien - finals_out.xlsx', index = False)
finals_train.to_excel(r'DNS_2/Phu kien - finals_train.xlsx', index = False)
#print(a.predict_on_dataframes())




