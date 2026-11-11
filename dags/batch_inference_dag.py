"""
DAG для batch-инференса недельных продаж сети дарксторов «Прилавок».

Пайплайн выполняет следующие задачи:
1. Загружает sales, plan, stores и features из PostgreSQL.
2. Создаёт признаки той же функцией, которая использовалась при обучении.
3. Загружает обученную модель CatBoost из S3.
4. Строит пакетный прогноз и обрабатывает аномальные предсказания.
5. Записывает результат в таблицу predictions в PostgreSQL.

Настройка Airflow Connection:
    Conn Id: postgres_sales_db
    Conn Type: Postgres
    Host, Port, Database, Login, Password: параметры учебной БД

Настройка Airflow Variables:
    s3_bucket_name — имя бакета
    s3_model_key — путь к pickle-модели
    s3_access_key — ключ доступа
    s3_secret_key — секретный ключ

Дополнительно можно задать s3_endpoint_url. По умолчанию используется
Yandex Object Storage: https://storage.yandexcloud.net.

Секреты подставляются только через Airflow Connection и Variables.
"""

from datetime import datetime, timedelta
import io
import logging
import os
import pickle
import tempfile

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import boto3
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import text

from preprocessing import HISTORY_FEATURES, preprocess_data


POSTGRES_CONN_ID = "postgres_sales_db"
logger = logging.getLogger(__name__)


def load_data_from_postgres(**context):
    """
    Загрузка данных для инференса из PostgreSQL.

    Исторические продажи до первой даты плана и будущие строки объединяются в
    таблице inference_data_temp. К ним присоединяются stores и features.
    Большой датафрейм не передаётся через XCom: туда сохраняется только первая
    дата плана, по которой следующая задача отделит строки для инференса.

    В лог выводятся количество строк, минимальная и максимальная даты.
    """
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    engine = hook.get_sqlalchemy_engine()

    sales = pd.read_sql(text("SELECT * FROM sales"), engine)
    stores = pd.read_sql(text("SELECT * FROM stores"), engine)
    features = pd.read_sql(text("SELECT * FROM features"), engine)
    plan = pd.read_sql(text("SELECT * FROM plan"), engine)

    for dataframe in (sales, features, plan):
        dataframe["date"] = pd.to_datetime(dataframe["date"])

    plan_first_date = plan["date"].min()
    sales = sales[sales["date"] < plan_first_date]
    plan["weekly_sales"] = np.nan

    inference_data = pd.concat([sales, plan], ignore_index=True, sort=False)
    inference_data = inference_data.merge(
        features,
        on=["store", "dept", "date"],
        how="left",
    )
    inference_data = inference_data.merge(stores, on="store", how="left")
    inference_data = inference_data.sort_values(["store", "dept", "date"])

    # Исходные таблицы после объединения больше не нужны. Освобождаем их перед
    # записью, чтобы снизить потребление памяти Airflow worker.
    del sales, stores, features, plan

    logger.info("Строк в inference_data_temp: %s", len(inference_data))
    logger.info("Минимальная дата: %s", inference_data["date"].min())
    logger.info("Максимальная дата: %s", inference_data["date"].max())
    logger.info("Начинаем пакетную запись inference_data_temp по 1000 строк")

    inference_data.to_sql(
        "inference_data_temp",
        engine,
        if_exists="replace",
        index=False,
        chunksize=1_000,
        method="multi",
    )
    logger.info("Таблица inference_data_temp успешно записана")
    context["ti"].xcom_push(
        key="plan_first_date",
        value=plan_first_date.strftime("%Y-%m-%d"),
    )


def preprocess_features(**context):
    """
    Предобработка признаков для инференса.

    1. Получает первую дату плана из XCom.
    2. Читает inference_data_temp из PostgreSQL.
    3. Выполняет preprocess_data.
    4. Оставляет строки плана и удаляет weekly_sales.
    5. Заполняет недоступные исторические признаки нулями.
    6. Удаляет строки с оставшимися пропусками.
    7. Передаёт подготовленный датафрейм через XCom в JSON.
    8. Удаляет временную таблицу inference_data_temp.
    """
    plan_first_date = pd.Timestamp(
        context["ti"].xcom_pull(key="plan_first_date", task_ids="load_data")
    )

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    engine = hook.get_sqlalchemy_engine()

    data = pd.read_sql(text("SELECT * FROM inference_data_temp"), engine)
    data = preprocess_data(data)
    inference_data = data[data["date"] >= plan_first_date].copy()

    # Для будущих недель фактических продаж ещё нет. Для новых пар
    # store-dept также может не быть средней истории, поэтому все
    # недоступные исторические признаки заполняем нулями.
    inference_data[HISTORY_FEATURES] = inference_data[HISTORY_FEATURES].fillna(0)
    inference_data = inference_data.drop(columns=["weekly_sales"])
    inference_data = inference_data.dropna()

    context["ti"].xcom_push(
        key="inference_data",
        value=inference_data.to_json(orient="split", date_format="iso"),
    )
    logger.info("Подготовлено строк для инференса: %s", len(inference_data))

    # Удаляем временную таблицу только после успешной предобработки и записи
    # результата в XCom. При ошибке таблица останется доступной для retry.
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS inference_data_temp"))


def load_model_from_s3(**context):
    """
    Загружает обученную CatBoost-модель из S3 через io.BytesIO.

    Модель сохраняется во временный файл, а путь передаётся через XCom.
    """
    s3_client = boto3.client(
        "s3",
        endpoint_url=Variable.get(
            "s3_endpoint_url",
            default_var="https://storage.yandexcloud.net",
        ),
        aws_access_key_id=Variable.get("s3_access_key"),
        aws_secret_access_key=Variable.get("s3_secret_key"),
        region_name="ru-central1",
    )

    model_buffer = io.BytesIO()
    s3_client.download_fileobj(
        Variable.get("s3_bucket_name"),
        Variable.get("s3_model_key", default_var="catboost_model.pkl"),
        model_buffer,
    )
    model_buffer.seek(0)

    run_id = context["run_id"].replace(":", "_").replace("+", "_")
    model_path = os.path.join(tempfile.gettempdir(), f"catboost_model_{run_id}.pkl")
    with open(model_path, "wb") as model_file:
        model_file.write(model_buffer.read())

    context["ti"].xcom_push(key="model_path", value=model_path)
    logger.info("Модель сохранена во временный файл %s", model_path)


def run_batch_inference(**context):
    """
    Выполняет batch-инференс.

    Получает датафрейм и путь к модели из XCom, загружает pickle-модель,
    выполняет predict, заменяет отрицательные и нечисловые прогнозы нулём и
    передаёт итоговый датафрейм через XCom в JSON.
    """
    inference_json = context["ti"].xcom_pull(
        key="inference_data",
        task_ids="preprocess_features",
    )
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="load_model")
    inference_data = pd.read_json(io.StringIO(inference_json), orient="split")

    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Финальная модель обучалась после PSI-фильтрации. Берём признаки и их
        # порядок прямо из модели, чтобы обучение и инференс не расходились.
        model_features = list(model.feature_names_)
        missing_features = sorted(set(model_features) - set(inference_data.columns))
        if missing_features:
            raise ValueError(f"Не хватает признаков модели: {missing_features}")

        predictions = model.predict(inference_data[model_features])
        predictions = np.asarray(predictions, dtype=float)
        predictions[~np.isfinite(predictions) | (predictions < 0)] = 0

        result = inference_data[["store", "dept", "date"]].copy()
        result["predicted_weekly_sales"] = predictions
        context["ti"].xcom_push(
            key="predictions",
            value=result.to_json(orient="split", date_format="iso"),
        )
        logger.info("Получено предсказаний: %s", len(result))
    finally:
        if model_path and os.path.exists(model_path):
            os.remove(model_path)


def save_predictions_to_postgres(**context):
    """
    Записывает прогнозы в таблицу predictions.

    Таблица пересоздаётся со столбцами store, dept, date,
    predicted_weekly_sales и prediction_timestamp. После вставки в лог
    выводятся количество строк и первые пять записей.
    """
    predictions_json = context["ti"].xcom_pull(
        key="predictions",
        task_ids="run_inference",
    )
    predictions = pd.read_json(io.StringIO(predictions_json), orient="split")
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.date

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    connection = hook.get_conn()

    try:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS predictions")
            cursor.execute(
                """
                CREATE TABLE predictions (
                    store INT,
                    dept INT,
                    date DATE,
                    predicted_weekly_sales FLOAT,
                    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (store, dept, date)
                )
                """
            )

            values = [
                (int(row.store), int(row.dept), row.date, float(row.predicted_weekly_sales))
                for row in predictions.itertuples(index=False)
            ]
            execute_values(
                cursor,
                """
                INSERT INTO predictions (store, dept, date, predicted_weekly_sales)
                VALUES %s
                ON CONFLICT (store, dept, date) DO UPDATE SET
                    predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
                    prediction_timestamp = CURRENT_TIMESTAMP
                """,
                values,
            )

            cursor.execute("SELECT COUNT(*) FROM predictions")
            rows_count = cursor.fetchone()[0]
            cursor.execute(
                """
                SELECT store, dept, date, predicted_weekly_sales, prediction_timestamp
                FROM predictions
                ORDER BY date, store, dept
                LIMIT 5
                """
            )
            first_rows = cursor.fetchall()
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    logger.info("Строк в таблице predictions: %s", rows_count)
    logger.info("Первые пять строк predictions: %s", first_rows)


default_args = {
    "owner": "student",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="sales_prediction_batch_inference",
    default_args=default_args,
    description="Batch-инференс недельных продаж для сети «Прилавок»",
    schedule_interval="0 20 * * 0",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sales", "ml", "batch-inference"],
) as dag:
    task_load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data_from_postgres,
    )
    task_preprocess = PythonOperator(
        task_id="preprocess_features",
        python_callable=preprocess_features,
    )
    task_load_model = PythonOperator(
        task_id="load_model",
        python_callable=load_model_from_s3,
    )
    task_inference = PythonOperator(
        task_id="run_inference",
        python_callable=run_batch_inference,
    )
    task_save_predictions = PythonOperator(
        task_id="save_predictions",
        python_callable=save_predictions_to_postgres,
    )

    task_load_data >> task_preprocess
    [task_preprocess, task_load_model] >> task_inference >> task_save_predictions
