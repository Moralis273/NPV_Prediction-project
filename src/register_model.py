import mlflow
from mlflow.tracking import MlflowClient
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Настройка клиента MLflow
    client = MlflowClient()
    
    # Поиск последнего запуска
    experiment = client.get_experiment_by_name("NPV_Prediction_DVC")
    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"])
    
    if runs:
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        # Регистрация модели
        model_name = f"{params['model']['name']}_NPV"
        
        try:
            # Пробуем найти существующую модель
            client.get_registered_model(model_name)
            print(f"Модель {model_name} уже существует")
        except:
            # Создаем новую зарегистрированную модель
            client.create_registered_model(model_name)
            print(f"Создана новая модель: {model_name}")
        
        # Добавляем версию
        client.create_model_version(
            name=model_name,
            source=f"mlruns/{experiment.experiment_id}/{run_id}/artifacts/model",
            run_id=run_id
        )
        
        print(f"✅ Модель {model_name} зарегистрирована в MLflow Model Registry")
    else:
        print("❌ Не найдены запуски для регистрации")

if __name__ == "__main__":
    main()