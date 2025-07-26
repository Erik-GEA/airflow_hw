import pandas as pd
import dill
import json
import glob
import os
from datetime import datetime


def get_latest_model_file(path):
    """Находит последний .pkl файл по времени модификации"""
    model_dir = f'{path}/data/models'
    pkl_files = glob.glob(f'{model_dir}/*.pkl')

    if not pkl_files:
        raise FileNotFoundError(f"Не найдено .pkl файлов в директории {model_dir}")

    # Находим файл с самым свежим временем модификации
    latest_file = max(pkl_files, key=os.path.getmtime)
    return latest_file

def predict():
    # Получаем путь до папки проекта
    path = os.environ.get('PROJECT_PATH', '.')

    # Находим и загружаем последнюю модель
    model_path = get_latest_model_file(path)
    print(f"Загружаем модель: {model_path}")

    with open(model_path, 'rb') as file:
        model = dill.load(file, ignore=True)

    # Создаем DataFrame для хранения всех предсказаний
    all_predictions = []

    # Готовим путь до файлов для теста
    path_files = f'{path}/data/test/*.json'

    # Перебираем тестовые файлы
    for json_files_path in glob.iglob(path_files):
        with open(json_files_path) as fin:
            form = json.load(fin)

            df = pd.DataFrame.from_dict([form])

            # Сохраняем car_id (предполагаю, что это поле id в данных)
            car_id = form.get('id', 'unknown')

            # Делаем предсказание
            pred = model.predict(df)

            # Добавляем в общий список предсказаний
            all_predictions.append({
                'car_id': car_id,
                'pred': pred[0]  # pred обычно возвращается как массив, берем первый элемент
            })

    # Создаем итоговый DataFrame со всеми предсказаниями
    df_pred = pd.DataFrame(all_predictions)

    # Сохраняем все предсказания в один файл
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    predictions_path = f'{path}/data/predictions/preds_{timestamp}.csv'
    df_pred.to_csv(predictions_path, index=False)

    print(f"Предсказания сохранены в: {predictions_path}")
    print(df_pred)


if __name__ == '__main__':
    predict()