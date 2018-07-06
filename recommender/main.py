# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:51:21 2018

@author: hungn
"""


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify
from flask import abort
nltk.download('stopwords')

from sklearn.metrics.pairwise import linear_kernel

def create_tfidf_matrix(file):
    tf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,3),max_df = 0.6, min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf_vect.fit_transform(file["Description"])
    return (tf_vect, tfidf_matrix)
    

def find_similar_in_same_matrix(tfidf_matrix, index, top_n = 3):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

def find_similar(file, tfidf_matrix, input_string,tf_vectorizer, top_n = 3):
    string_vector = tf_vectorizer.transform(np.array([input_string]))
    cosine_similarities = linear_kernel(string_vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    top_results = [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]
    
    for (index, score) in top_results:
        i = 0
        result = file.iloc[index]
        print("Score: " + str(score))
        for column in result:
            print(file.columns.values[i] + ": \t" + column)
            i += 1
        print("\n")
        
def get_recommendation(input_string, configuration):
    """
       CONFIGURATION: an array of tuples, each tuple contain 3 elements
       describing the corresponding tier name, output file, tfidf_matrix and tf_vectorizer
       
       Example: [(name, it_towers, tfidf_matrix_it_towers, tf_it_towers), (..,..,..),..]
    """
    
    print("Input String: " + input_string)
    for name, file, tfidf_matrix, tf_vectorizer in configuration:
        print("=======Tier " + str(name) + ": =========")
        find_similar(file=file,
                     tfidf_matrix=tfidf_matrix, 
                     input_string=input_string,
                     tf_vectorizer=tf_vectorizer
                     )


def find_similar_json(filename, file, tfidf_matrix, input_string,tf_vectorizer, top_n = 3):
    """

    :param filename:
    :param file:
    :param tfidf_matrix:
    :param input_string:
    :param tf_vectorizer:
    :param top_n:
    :return: JSON Sample
    {
        "Tier" : filename,
        "Results": [
            {
                "Score": ,
                "..." : ,
                "Description" :

            },
            {
            "Score": ,
                "..." : ,
                "Description" :

            },
            {
            "Score": ,
                "..." : ,
                "Description" :
            }
        ]
    }
    """



    string_vector = tf_vectorizer.transform(np.array([input_string]))
    cosine_similarities = linear_kernel(string_vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    top_results = [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    output = {
        "Tier": filename,
        "Results": []
    }
    count = 0
    for (index, score) in top_results:

        i = 0
        result = file.iloc[index]
        print("Score: " + str(score))
        output["Results"].append({
            "Score" : score
        })
        for column in result:
            output["Results"][count][file.columns.values[i]] = column
            print(file.columns.values[i] + ": \t" + column)
            i += 1
        print("\n")
        count += 1
    return output

        
    
    
    
#Read 3 files
it_towers = pd.read_excel("IT_Towers.xlsx")
services = pd.read_excel("Services.xlsx")
cost_pools = pd.read_excel("CostPools.xlsx")

#Test data
""" No test data yet. But should able to use the real interaction of user
    to create test data and verify the accuracy of the model"""

#***********************************IT TOWERS MODEL***************************
#Build the model from sklearn
tf_it_towers, tfidf_matrix_it_towers = create_tfidf_matrix(it_towers)
#***********************************SERVICES MODEL***************************
tf_services, tfidf_matrix_services = create_tfidf_matrix(services)
#***********************************COST POOLS MODEL***************************
tf_cost_pools, tfidf_matrix_cost_pools = create_tfidf_matrix(cost_pools)

#Find similar
find_similar(file=it_towers,
             tfidf_matrix=tfidf_matrix_it_towers,
             input_string= "My database has troubled connecting to the server",
             tf_vectorizer=tf_it_towers)    

    
#result
"""
    IO TEST FOR METHOD


"""
"""while True:
    input_string = input("Enter your description to classify: ")
    get_recommendation(input_string=input_string,
                   configuration=[
                           ("IT Towers",
                            it_towers,
                            tfidf_matrix_it_towers,
                            tf_it_towers,
                                   ),
                            ("Services",
                            services,
                            tfidf_matrix_services,
                            tf_services,
                                   ),
                            ("Cost Pools",
                            cost_pools,
                            tfidf_matrix_cost_pools,
                            tf_cost_pools,
                                   )
                           ])
                            
      """


# jsonoutput = find_similar_json("IT Towers", it_towers, tfidf_matrix_it_towers, "Company needs financial aid for database infrastructure", tf_it_towers, top_n = 3)
# print(jsonoutput)

#======================FLASK API SERVICE ====================================


app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': "Use Post method to get recommendation ",
        'description': "JSON input with string to recommend with description property",
    }
]

@app.route('/', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

from flask import request
@app.route('/home', methods=['POST'])
def create_task():
    if not request.json or not 'description' in request.json:
        abort(400)
    input_string = request.json['description']
    results = []
    results.append(find_similar_json("IT Towers",
                                     it_towers,
                                     tfidf_matrix_it_towers,
                                     input_string,
                                     tf_it_towers,
                                     top_n=3))
    results.append(find_similar_json("Services",
                                     services,
                                     tfidf_matrix_services,
                                     input_string,
                                     tf_services,
                                     top_n=3))
    results.append(find_similar_json("Cost Pools",
                                     cost_pools,
                                     tfidf_matrix_cost_pools,
                                     input_string,
                                     tf_cost_pools,
                                     top_n=3))

    return jsonify({'data': results}), 201
if __name__ == '__main__':
    app.run(debug=True)

