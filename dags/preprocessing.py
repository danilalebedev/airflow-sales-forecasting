"""
Функции предобработки данных для обучения и batch-инференса.

В этот файл вынесены функции из Jupyter Notebook. Функция preprocess_data:
1. Обрабатывает аномальные продажи.
2. Заполняет пропуски в признаках средними значениями.
3. Обогащает датасет признаками с помощью функций:
   create_temporal_features,
   create_avg_sales_feature,
   create_lag_features,
   create_rolling_features.

Пропуски weekly_sales не заполняются, потому что ими обозначены будущие строки
из таблицы plan. Иначе в лаговые признаки попало бы искусственное значение.
"""

import pandas as pd


GROUP_COLUMNS = ["store", "dept"]
HISTORY_FEATURES = [
    "avg_sales_before",
    "sales_1week_ago",
    "sales_2week_ago",
    "sales_4week_ago",
    "mean_sales_2week",
    "mean_sales_4week",
]


def create_temporal_features(df):
    """Добавляет календарные признаки из даты."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    return df


def create_avg_sales_feature(df):
    """Считает средние прошлые продажи без утечки целевого значения."""
    df = df.sort_values(GROUP_COLUMNS + ["date"]).copy()
    grouped_sales = df.groupby(GROUP_COLUMNS, sort=False)["weekly_sales"]
    df["avg_sales_before"] = grouped_sales.transform(
        lambda values: values.shift(1).expanding().mean()
    )
    return df


def create_lag_features(df):
    """Добавляет продажи за 1, 2 и 4 предыдущие недели."""
    df = df.sort_values(GROUP_COLUMNS + ["date"]).copy()
    grouped_sales = df.groupby(GROUP_COLUMNS, sort=False)["weekly_sales"]
    df["sales_1week_ago"] = grouped_sales.shift(1)
    df["sales_2week_ago"] = grouped_sales.shift(2)
    df["sales_4week_ago"] = grouped_sales.shift(4)
    return df


def create_rolling_features(df):
    """Добавляет средние продажи за 2 и 4 предыдущие недели."""
    df = df.sort_values(GROUP_COLUMNS + ["date"]).copy()
    grouped_sales = df.groupby(GROUP_COLUMNS, sort=False)["weekly_sales"]
    df["mean_sales_2week"] = grouped_sales.transform(
        lambda values: values.shift(1).rolling(2).mean()
    )
    df["mean_sales_4week"] = grouped_sales.transform(
        lambda values: values.shift(1).rolling(4).mean()
    )
    return df


def preprocess_data(df):
    """
    Полная предобработка данных, идентичная логике обучения.

    Отрицательные продажи заменяются нулём. Пропуски числовых признаков
    заполняются средним, категориальных — модой. После очистки создаются
    календарные, средние, лаговые признаки и скользящие средние.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates()

    # Отрицательные продажи считаем аномальными. Пропуски в целевом столбце
    # сохраняем: ими отмечены строки плана.
    df["weekly_sales"] = df["weekly_sales"].clip(lower=0)

    numeric_columns = df.select_dtypes(include="number").columns.difference(
        ["weekly_sales"]
    )
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        if df[column].isna().any():
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    return df
