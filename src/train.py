import joblib
import json
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import yaml
import mlflow

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Загрузка данных
    print("Загрузка обработанных данных...")
    data = joblib.load(params['data']['processed_path'])
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    # Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("NPV_Prediction_DVC")
    
    with mlflow.start_run():
        # Обучение модели
        print("Обучение модели...")
        model = XGBRegressor(
            random_state=params['preprocessing']['random_state'],
            objective='reg:squarederror',
            **params['model']['hyperparameters']
        )
        
        model.fit(X_train, y_train)
        
        # Кросс-валидация
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=params['training']['cv_folds'],
            scoring=params['training']['scoring']
        )
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Логирование в MLflow
        mlflow.log_params(params['model']['hyperparameters'])
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        mlflow.sklearn.log_model(model, "model")
        
        # Сохранение модели и метрик
        joblib.dump(model, 'models/model.joblib')
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Модель обучена. R2: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()