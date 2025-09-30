import json
import yaml
import pandas as pd
from datetime import datetime

def generate_report():
    # Загрузка метрик и параметров
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    with open('models/evaluation.json', 'r') as f:
        evaluation = json.load(f)
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Создание отчета
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'name': params['model']['name'],
            'hyperparameters': params['model']['hyperparameters']
        },
        'performance': {
            'cross_validation': {
                'mean_r2': metrics['cv_mean'],
                'std_r2': metrics['cv_std']
            },
            'test_set': evaluation['test_metrics'],
            'predictions_quality': evaluation['predictions_stats']
        },
        'data_info': {
            'target': params['features']['target'],
            'features_count': len(params['features']['drop_columns']) + 1  # приблизительно
        }
    }
    
    # Сохранение отчета
    with open('reports/model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📊 Отчет создан:")
    print(f"   Модель: {report['model_info']['name']}")
    print(f"   CV R²: {report['performance']['cross_validation']['mean_r2']:.4f} ± {report['performance']['cross_validation']['std_r2']:.4f}")
    print(f"   Test R²: {report['performance']['test_set']['r2']:.4f}")

if __name__ == "__main__":
    generate_report()