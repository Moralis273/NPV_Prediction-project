import joblib
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Загрузка данных и модели
    data = joblib.load(params['data']['processed_path'])
    model = joblib.load('models/model.joblib')
    
    X_test, y_test = data['X_test'], data['y_test']
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Детальная оценка
    evaluation = {
        'test_metrics': {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        },
        'predictions_stats': {
            'actual_mean': float(y_test.mean()),
            'predicted_mean': float(y_pred.mean()),
            'actual_std': float(y_test.std()),
            'predicted_std': float(y_pred.std())
        },
        'residuals_analysis': {
            'residuals_mean': float((y_test - y_pred).mean()),
            'residuals_std': float((y_test - y_pred).std())
        }
    }
    
    # Сохранение детальной оценки
    with open('models/evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print("✅ Детальная оценка завершена")
    print(f"R² на тесте: {evaluation['test_metrics']['r2']:.4f}")

if __name__ == "__main__":
    main()