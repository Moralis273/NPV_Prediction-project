import mlflow
from mlflow.tracking import MlflowClient
import yaml
import json
import os
from datetime import datetime

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Настройка клиента MLflow
    client = MlflowClient()
    
    try:
        # Поиск последнего запуска
        experiment = client.get_experiment_by_name("NPV_Prediction_DVC")
        if not experiment:
            print("❌ Эксперимент не найден")
            # Создаем файл с ошибкой
            registry_info = {"error": "Experiment not found"}
            os.makedirs('registry', exist_ok=True)
            with open('registry/model_info.json', 'w') as f:
                json.dump(registry_info, f, indent=2)
            return
            
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
                model_status = "existing"
            except:
                # Создаем новую зарегистрированную модель
                client.create_registered_model(model_name)
                print(f"Создана новая модель: {model_name}")
                model_status = "new"
            
            # Добавляем версию
            model_version = client.create_model_version(
                name=model_name,
                source=f"mlruns/{experiment.experiment_id}/{run_id}/artifacts/model",
                run_id=run_id
            )
            
            # Сохраняем информацию о регистрации
            registry_info = {
                "model_name": model_name,
                "model_version": model_version.version,
                "run_id": run_id,
                "status": model_status,
                "timestamp": datetime.now().isoformat()  # ИСПРАВЛЕНО!
            }
            
            # Создаем папку registry если не существует
            os.makedirs('registry', exist_ok=True)
            
            # Сохраняем информацию в файл
            with open('registry/model_info.json', 'w') as f:
                json.dump(registry_info, f, indent=2)
            
            print(f"✅ Модель {model_name} зарегистрирована в MLflow Model Registry")
            print(f"📁 Информация сохранена в registry/model_info.json")
            
        else:
            print("❌ Не найдены запуски для регистрации")
            # Создаем файл с ошибкой
            os.makedirs('registry', exist_ok=True)
            with open('registry/model_info.json', 'w') as f:
                json.dump({"error": "No runs found"}, f, indent=2)
                
    except Exception as e:
        print(f"❌ Ошибка при регистрации модели: {e}")
        # Создаем файл с информацией об ошибке
        os.makedirs('registry', exist_ok=True)
        with open('registry/model_info.json', 'w') as f:
            json.dump({"error": str(e)}, f, indent=2)

if __name__ == "__main__":
    main()