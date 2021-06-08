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

#Initial training process
#df= pd.read_excel('../finals.xlsx', sheet_name='Sheet1')

df= pd.read_excel('getir.xlsx', sheet_name='getir')

model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
        beta_2=0.999, early_stopping=False, epsilon=1e-08,
        hidden_layer_sizes=(100,50,10), learning_rate='constant',
        learning_rate_init=0.001, max_iter=1000, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=None,
        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
        verbose=False, warm_start=False)

#pkl_filename = "pickle__model.pkl"

#with open(pkl_filename, 'rb') as file:
#    pickle_model = pickle.load(file)

def process_dataframe(df):
   df['uniqueNumberCount'] = df.apply(fe.get_unique_number_count, axis=1)
   df['numberMatchRate'] = df.apply(fe.get_rate, axis=1)
   df['matchScore'] = df.apply(fe.sorted_levenshtein_rate_apply, axis=1)
   df['normalizedMatchRate'] = (df['numberMatchRate']+2).apply(np.log)
   df['squaredPriceRate'] = df['ti_le_gia']**2
   return df

df = process_dataframe(df)

X = df[['matchScore', 'squaredPriceRate', 'uniqueNumberCount', 'normalizedMatchRate']].values
y = df['Match'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify = y)

#sc = StandardScaler()
#sc.fit(X)
#X_train_std = sc.transform(X)
#X_test_std = sc.transform(X)

pipeline = make_pipeline(StandardScaler(), model)

pipeline.fit(X_train, y_train)
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)
#y_pred = pickle_model.predict(X_test_std)
#df['Match'] = y_pred
#print(y_pred)

#df.to_excel(r'finals/finals1.xlsx', index = False)
preds = pipeline.predict(X_test)
print(preds)
print('Accuracy: %.2f' % accuracy_score(y_test, preds))

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
        result = pd.DataFrame(columns=('id','id_match','product1', 'product2', 'gia', 'gia2','ti_le_gia', 'phanloai'))
        for index, product in self.df.iterrows():
            temp = {'id': self.row['id'],
            'id_match': product['id'],
            'product1': str(self.row['name']), 
            'product2': str(product['name']),
            'gia': self.row['price'],
            'gia2': product['price'],
            'ti_le_gia': self.row['price']/product['price'],
            'phanloai': self.row['phanloai']
            }
            #print(temp)
            result = result.append(temp, ignore_index = True)
        #print(result.shape[0])
        return result

class ProcessAndPredict:
    def __init__(self, row_in, df_out):
        self.row_in = row_in
        self.df_out = df_out 
    def create_dataframe_and_predict(self):
        otm = OneToMany(self.row_in, self.df_out)
        process_otm = otm.createOneToManyDataframe()
        process_otm_size = self.df_out.shape[0]
        rd = process_dataframe(process_otm)[['matchScore', 'squaredPriceRate', 'uniqueNumberCount', 'normalizedMatchRate']].values
        process_otm['match'] = pipeline.predict(rd)
        return (process_otm_size, process_otm['match'])

class FindFirstAndAssign:
    def __init__(self, process_otm, df_out):
        self.process_otm = process_otm
        self.df_out = df_out
    def find_matches_and_assign(self):
        match_counter = 0
        for index, result in self.process_otm.iterrows():
            match_counter += 1

            if(result.match == 1):
                    #print(result)
                final_temp = {
                    'id': result.id,
                    'id_match': result.id_match,
                    'name': result.product1,
                    'price': result.gia,
                    'phanloai': result.phanloai,
                }
                #print(final_temp)

                    #print(final_temp['ten'] + ' ---- ' + final_temp['product_out'])                   
                self.df_out = self.df_out.append(final_temp, ignore_index = True)
                break
                
            if(match_counter == self.process_otm_size):
                final_temp = {
                    'id': result.id,
                    'id_match': result.id,
                    'name': result.product1,
                    'price': result.gia,
                    'phanloai': result.phanloai
                    }
#                print(final_temp)

                self.df_out = self.df_out.append(final_temp, ignore_index = True)
        return self.df_out

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
            #print(process_otm)
            #process_otm_size = self.df_out.shape[0]
            #print(list(process_otm.columns))

            #Process the dataframe and collect the results
            rd = process_dataframe(process_otm)[['matchScore', 'squaredPriceRate', 'uniqueNumberCount', 'normalizedMatchRate']].values
            #print(rd)
            #sc.fit(rd)
            #rd_std = sc.transform(rd)
            process_otm['match'] = pipeline.predict(rd)
            #print(process_otm[['match', 'squaredPriceRate']])
            #Add the results as a new colum in OneToMany dataframe
            #print(process_otm[['product1','product2','match']])
            #0.7,
            condition = (process_otm.match == 1) & (process_otm.matchScore >= 0.68) & (process_otm.uniqueNumberCount <= 4)
            p = process_otm[condition]
            
            #print(process_otm.empty)
            #print(p.empty)
            if(p.empty):
                #print(row_in) 
                final_temp = {
                        'id': row_in.id,
                        'id_match': row_in.id,
                        'name': row_in['name'],
                        'price': row_in.price,
                        'phanloai': row_in.phanloai
                }
                print('not matched')
                print(final_temp)
                self.df_out = self.df_out.append(final_temp, ignore_index = True)
            else:
                #print(p['squaredPriceRate'].sub(1).abs())
                pos = p['squaredPriceRate'].sub(1).abs().idxmin()
                #print(pos)
                result = process_otm.loc[pos]
                print('matched')
                print('matchScore: ' + str(result['matchScore']))
                print('uniqueNumberCount: '+ str(result['uniqueNumberCount'])) 
                final_temp = {
                        'id': result.id,
                        'id_match': result.id_match,
                        'name': result.product1,
                        'price': result.gia,
                        'phanloai': result.phanloai,
                    }

                train_temp = {
                        'product1': result.product1,
                        'product2': result.product2,
                        'gia': result.gia,
                        'gia2': result.gia2,
                        'ti_le_gia': result.ti_le_gia,
                        'Match': 1
                    }
                    #print(final_temp)
                print(train_temp)
                    #print(final_temp['ten'] + ' ---- ' + final_temp['product_out'])                   
                self.df_out = self.df_out.append(final_temp, ignore_index = True)
                self.df_train = self.df_train.append(train_temp, ignore_index = True)
            #match count
            #match_counter = 0
            # for index, result in process_otm.iterrows():
            #     match_counter += 1

            #     if(result.match == 1):
            #         #print(result)
            #         final_temp = {
            #             'id': result.id,
            #             'id_match': result.id_match,
            #             'name': result.product1,
            #             'price': result.gia,
            #             'phanloai': result.phanloai,
            #         }

            #         train_temp = {
            #             'product1': result.product1,
            #             'product2': result.product2,
            #             'gia': result.gia,
            #             'gia2': result.gia2,
            #             'ti_le_gia': result.ti_le_gia,
            #             'Match': 1
            #         }
            #         #print(final_temp)
            #         print(train_temp)
            #         #print(final_temp['ten'] + ' ---- ' + final_temp['product_out'])                   
            #         self.df_out = self.df_out.append(final_temp, ignore_index = True)
            #         self.df_train = self.df_train.append(train_temp, ignore_index = True)
            #         break
                
            #     if(match_counter == process_otm_size):
            #         final_temp = {
            #             'id': result.id,
            #             'id_match': result.id,
            #             'name': result.product1,
            #             'price': result.gia,
            #             'phanloai': result.phanloai
            #             }
            #         #print(final_temp)

            #         # train_temp = {
            #         #     'product1': result.product1,
            #         #     'product2': result.product2,
            #         #     'gia': result.gia,
            #         #     'gia2': result.gia2,
            #         #     'ti_le_gia': result.ti_le_gia,
            #         #     'Match': 0
            #         # }
            #         # #print(final_temp)
            #         # print(train_temp)
            #         # self.df_train = self.df_train.append(train_temp, ignore_index = True)

            #         self.df_out = self.df_out.append(final_temp, ignore_index = True)

                    #final_results = final_results.append(final_temp, ignore_index = True)
        return self.df_out, self.df_train


df_train = pd.DataFrame()

a = Predictions(df_in, df_out, df_train)
#a.predict_on_dataframes()
finals_out, finals_train = a.predict_on_dataframes()
#print(finals.df_out)
finals_out.to_excel(r'DNS_2/Phu kien - finals_out.xlsx', index = False)
finals_train.to_excel(r'DNS_2/Phu kien - finals_train.xlsx', index = False)
#print(a.predict_on_dataframes())




