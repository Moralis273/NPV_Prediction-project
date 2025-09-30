import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml
import os

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_excel(params['data']['raw_path'])
    
    # Предобработка
    print("Предобработка данных...")
    X = df.drop(params['features']['drop_columns'] + [params['features']['target']], axis=1)
    y = df[params['features']['target']]
    
    # Кодирование категориальных переменных
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cols = encoder.fit_transform(X[params['features']['categorical_columns']])
    encoded_df = pd.DataFrame(
        encoded_cols, 
        columns=encoder.get_feature_names_out(params['features']['categorical_columns'])
    )
    
    X_processed = pd.concat([X.drop(params['features']['categorical_columns'], axis=1), encoded_df], axis=1)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, 
        test_size=params['preprocessing']['test_size'], 
        random_state=params['preprocessing']['random_state']
    )
    
    # Сохранение
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump({
        'X_train': X_train, 'X_test': X_test, 
        'y_train': y_train, 'y_test': y_test,
        'feature_names': X_processed.columns.tolist()
    }, params['data']['processed_path'])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/encoder.joblib')
    joblib.dump(X_processed.columns.tolist(), 'models/feature_columns.joblib')
    
    print("✅ Данные успешно предобработаны")

if __name__ == "__main__":
    main()