from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NPV Prediction API", version="1.0")

# Загрузка модели и энкодера из DVC-версионированных файлов
try:
    model = joblib.load('models/model.joblib')
    encoder = joblib.load('models/encoder.joblib')
    feature_columns = joblib.load('models/feature_columns.joblib')
    logger.info("Модель и энкодер успешно загружены")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    # Модель может быть не обучена - это нормально для разработки
    model = None
    encoder = None
    feature_columns = None

# Модель входных данных
class InputData(BaseModel):
    Heff: float = Field(..., ge=0, description="Эффективная толщина")
    Perm: float = Field(..., ge=0, description="Проницаемость")
    Sg: float = Field(..., ge=0, le=1, description="Газонасыщенность")
    L_hor: float = Field(..., ge=0, description="Горизонтальная длина")
    GS: str = Field(..., description="Тип ствола")
    temp: float = Field(..., ge=0, description="Темп падения")
    C5: float = Field(..., ge=0, description="Содержание C5")
    GRP: int = Field(..., ge=0, description="ГРП")
    nGS: int = Field(..., ge=0, description="Количество стволов")

@app.get("/")
async def root():
    return {"message": "NPV Prediction API", "status": "active", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Запустите пайплайн обучения.")
    
    try:
        input_dict = data.dict()
        
        # Создаем DataFrame с правильным порядком изначально
        numeric_data = {k: v for k, v in input_dict.items() if k != 'GS'}
        
        # Кодируем GS
        gs_encoded = encoder.transform([[input_dict['GS']]])
        gs_columns = encoder.get_feature_names_out(['GS'])
        gs_data = dict(zip(gs_columns, gs_encoded[0]))
        
        # Объединяем данные
        all_data = {**numeric_data, **gs_data}
        
        # Создаем DataFrame с правильным порядком
        input_processed = pd.DataFrame([all_data])[feature_columns]
        
        # Предсказание
        prediction = model.predict(input_processed)
        result = float(prediction[0])
        
        return {"predicted_NPV": round(result, 2), "status": "success"}
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Информация о загруженной модели"""
    try:
        features = []
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_.tolist()
        elif hasattr(model, 'get_booster'):
            features = model.get_booster().feature_names
        
        return {
            "model_type": type(model).__name__,
            "n_features": len(features),
            "features": features
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)