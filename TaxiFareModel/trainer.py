# imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
from sklearn.preprocessing import OneHotEncoder, StandardScaler

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = '[GB][LON][Al_Catto]test model version 1'

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        #take in X and y then move these to instance variables so you can use across methods  
        self.pipeline = None
        self.X = X
        self.y = y

    
   
          
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        return self.pipe

  


    def run(self):
        """set and train the pipeline"""
        # This will run the fit on the pipeline
        self.set_pipeline()
        self.mlflow_log_param('model', 'linear')
        self.pipe.fit(self.X_train, self.y_train)
        
      

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        final_rmse = compute_rmse(y_pred, y_test)

        return final_rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)




if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data()
    # set X and y
    y = df.pop("fare_amount")
    X = df
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trained = Trainer(X_train, y_train)
    # train
    trained.run()
    # evaluate
    trained.evaluate(X_test, y_test)
    print('TODO')
