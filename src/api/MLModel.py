import mlflow.pyfunc

class WeatherModel:
    def __init__(self, pyfunc_model):
        self._model = pyfunc_model

    @classmethod
    def load_from_registry(cls, model_name: str):
        uri = f"models:/{model_name}/Production"
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        return cls(pyfunc_model)

    def predict(self, df):
        return self._model.predict(df)
