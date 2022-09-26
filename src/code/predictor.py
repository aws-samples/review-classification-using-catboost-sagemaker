# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback
import csv
import flask
import time
import pandas as pd
import numpy as np
import copy
from catboost import CatBoostClassifier, Pool as CatboostPool, cv
from pandas.api.types import CategoricalDtype

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

all_cols = ['marketplace', 'customer_id', 'review_id', 'product_id', 'helpful_votes', 'verified_purchase',
            'review_headline', 'review_body', 'product_title', 'target']
numerical_cols = ['helpful_votes']
categorical_cols = ['verified_purchase']

text_cols = ['review_headline', 'review_body', 'product_title']


def ret_pool_obj(X):
    pool_obj = CatboostPool(
        data=X,
        label=None,
        text_features=text_cols,
        cat_features=categorical_cols,
        feature_names=list(X.columns)
    )
    return pool_obj


def pre_process(df):
    df.fillna(
        value={'review_headline': '', 'review_body': '', 'product_title': ''}, inplace=True)
    df.fillna(
        value={'verified_purchase': 'Unk'}, inplace=True)

    df.fillna(-9999, inplace=True)
    return df

    # Convert object columns to pandas categorical


def pre_process_cat(df, model):
    for col_name in categorical_cols:
        # Exclude asin column from converting to categorical
        # print(col_name)
        cat_type = CategoricalDtype(categories=model[0].get(col_name), ordered=False)
        df[col_name] = df[col_name].astype(cat_type, copy=False)
        df[col_name].fillna('Unk', inplace=True)

    col_names = df.columns
    category_col_names = col_names[df.dtypes == 'category']
    # Fill missing values in new data (categories column) with -9999
    for category in category_col_names:
        df[category].fillna('Unk', inplace=True)
    return df


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            print('i am here')
            model_file = CatBoostClassifier()
            model_file.load_model(os.path.join(model_path, 'model-classification-prod'))

            with open(os.path.join(model_path, 'obj_col_categories.pkl'), 'rb') as inp:
                obj_col_categories = pickle.load(inp)
                print('Model is loaded:-')
            cls.model = [obj_col_categories, model_file]
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = cls.get_model()
        print('Running inference:-')
        input_processed = pre_process_cat(input, model)
        input_pool = ret_pool_obj(input_processed[numerical_cols + categorical_cols + text_cols])

        # dvalid = clf[0].
        prob = model[1].predict_proba(input_pool)[:, 1]
        # print(prob)
        print('Complete inferencing on', prob.shape, ' records.')
        input['score'] = prob
        return input[['review_id', 'target', 'score']]


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    start = time.time()
    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        # print(flask.request.data)
        data = flask.request.data.decode('utf-8')
        # print(len(data))
        print('I am here bby')
        print('length of all_Cols', len(all_cols))
        data = pd.read_csv(io.StringIO(data), names=all_cols, sep='\t', lineterminator='\n', escapechar='\\',
                           quotechar='"', quoting=csv.QUOTE_NONE, keep_default_na=False)

        print('Column type: (Before Feature Engineering) ', data.dtypes)
        data = pre_process(data)
        print('After pre-processing shape of the data: ', data.shape)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Drop first column, since sample notebook uses train data to show case predictions
    # Do the prediction
    predictions = ScoringService.predict(data)
    print('Shape of predictions :', predictions.shape)
    print('columns of predictions :', predictions.columns)
    # Convert from numpy back to CSV
    out = io.StringIO()
    predictions.to_csv(out, header=False, index=False, sep='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_NONE)
    result = out.getvalue()
    end = time.time()

    print('Time to execute recs: ', predictions.shape, ' in time :', end - start)
    return flask.Response(response=result, status=200, mimetype='text/csv')
