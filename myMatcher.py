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
import pickle

df= pd.read_excel('getir.xlsx', sheet_name='getir')

model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
        beta_2=0.999, early_stopping=False, epsilon=1e-08,
        hidden_layer_sizes=(25, 25, 20), learning_rate='constant',
        learning_rate_init=0.001, max_iter=1000, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=None,
        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
        verbose=False, warm_start=False)

#pkl_filename = "pickle__model.pkl"

#with open(pkl_filename, 'rb') as file:
#    pickle_model = pickle.load(file)

def process_dataframe(df):
   df['uniqueNumberCount'] = df.apply(fe.get_unique_number_count, axis=1)+1
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

#--------------------------------------------------------

df_in = pd.read_excel('DNS/Hang tieu dung - input - 200.xlsx', sheet_name='Sheet1')

df_out = pd.read_excel('DNS/Hang tieu dung - output - 200.xlsx', sheet_name='Sheet1')

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

class Predictions:
    def __init__(self, df_in, df_out):
        self.df_in = df_in
        self.df_out = df_out
    def predict_on_dataframes(self):
        #new_product_id = 0

        for index, row_in in self.df_in.iterrows():
            otm = OneToMany(row_in, self.df_out)
            process_otm = otm.createOneToManyDataframe()
            process_otm_size = self.df_out.shape[0]

            rd = process_dataframe(process_otm)[['matchScore', 'squaredPriceRate', 'uniqueNumberCount', 'normalizedMatchRate']].values
            process_otm['match'] = pipeline.predict(rd)

            match_counter = 0

            for index, result in process_otm.iterrows():
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
                    print(final_temp)

                    #print(final_temp['ten'] + ' ---- ' + final_temp['product_out'])                   
                    self.df_out = self.df_out.append(final_temp, ignore_index = True)
                    break
                
                if(match_counter == process_otm_size):
                    final_temp = {
                        'id': result.id,
                        'id_match': result.id,
                        'name': result.product1,
                        'price': result.gia,
                        'phanloai': result.phanloai
                        }
                    print(final_temp)

                    self.df_out = self.df_out.append(final_temp, ignore_index = True)

                    #final_results = final_results.append(final_temp, ignore_index = True)
        return self.df_out

a = Predictions(df_in, df_out)
finals = a.predict_on_dataframes().to_excel(r'DNS/Hang tieu dung - final.xlsx', index = False)




