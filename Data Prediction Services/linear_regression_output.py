from flask import Flask, redirect, url_for, request, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import operator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_score
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='template')

@app.route('/auto_linear_regression_output', methods = ['GET','POST'])
def auto_linear_regression():

   global url
   global columninput

   url = request.form['url']
   columninput = request.form['columninput']
   num_configs = int(request.form['configurations'])
   num_rounds = int(request.form['rounds'])

   def model_predict():

      df = pd.read_csv(url)
      df_sample = df.sample(frac=0.05)
      y = df[columninput]
      y_train = df_sample[columninput]
      del df[columninput]
      del df_sample[columninput]
      x = df
      x_train = df_sample
      #Define the number of initial randomized layer configs and training rounds

      best_model = None
      best_loss = float('inf')

      for round in range(num_rounds):
          print(f"Round {round+1}:")
          #Generate random layer configs around the best model
          if round == 0:
              configs = []
              for _ in range(num_configs):
                  num_layers = np.random.randint(2, 6)
                  config = np.random.randint(5, 51, size=num_layers)
                  print("config: ", config)
                  configs.append(config)
          else:
              #Generate new configs around the best model from the previous round
              configs = []
              for _ in range(num_configs):
                  #Add or subtract random values around each layer in the best model
                  new_config = []
                  tuner = np.random.randint(-5, 6)
                  print("tuner val: ", tuner)
                  for i in best_config:
                      new_units = i + tuner
                      new_config.append(new_units)
                  configs.append(new_config)
              print("new config: ", configs)
          #Iterate over the layer configs
          for config in configs:
              model = keras.Sequential()
              model.add(layers.Dense(config[0], activation='relu', input_shape=(x_train.shape[1],)))
              for units in config[1:]:
                  model.add(layers.Dense(units, activation='relu'))
              model.add(layers.Dense(1))
              #Compile
              model.compile(optimizer='adam', loss='mse')
              #Train
              model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0)
              #Evaluate
              loss = model.evaluate(x_train, y_train)
              #Update the best model
              if round == 0:
                  best_config = config
              if loss < best_loss:
                  best_loss = loss
                  best_model = model
                  best_config = config
              print("Model:")
              print(model.summary())
              print("Loss:", loss)
              print("")
              print("")
              print("Best Model:")
              print(best_model.summary())
              print("Best Loss:", best_loss)
              print("Best Config:", best_config)
              print("")
              print("")
      print("Best Model:")
      print(best_model.summary())
      print("Best Loss:", best_loss)
      print("Best Config:", best_config)

      best_model.save('instance/best_regression_model.h5')
      return render_template('linear_regression_output.html', model=best_model.summary(), score=best_loss)
   
   return model_predict()

@app.route('/UPLOAD_FOLDER/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    os.makedirs(os.path.join(app.instance_path, 'jic files'), exist_ok=True)
    f.save(os.path.join(app.instance_path, 'jic files', secure_filename(f.filename)))

    return send_from_directory(directory=f, path=filename, as_attachment=True)

if __name__ == '__main__':
   app.run(debug = True, use_reloader=False)

#/Users/Yash/Desktop/Data Prediction Services/Data Prediction Services/jic files/UPLOAD_FOLDER