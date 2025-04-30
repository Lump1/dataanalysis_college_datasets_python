import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import loaddata as ld
import pandas as pd
import statsmodels.tsa.stattools as stattools
import calmap
import plotly.express as px
from gapminder import gapminder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def task_1(dataUrl: str) -> str:
    sns.set_style("whitegrid")

    # Load dataset
    dataset = ld.loadDataset(dataUrl)

    print(dataset.columns)

    if 'bmi' in dataset.columns and 'charges' in dataset.columns:
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(
            data=dataset, x="bmi", y="charges", hue="smoker",
            palette={"yes": "red", "no": "blue"}, alpha=0.6
        )
        scatter.ax
        plt.title("Залежність між ІМТ та медичними витратами з урахуванням статусу курця")
        plt.xlabel("Індекс маси тіла (BMI)")
        plt.ylabel("Медичні витрати (USD)")
        plt.legend(title="Куріння")

    else:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=dataset, x="Date", y="Value", hue="Sex")
        plt.title("Залежність між Датою і Значенням по Статтю")
        plt.xlabel("Дата")
        plt.ylabel("Значення")
        plt.legend(title="Стать")

    image_path = "task_1_plot.png"
    plt.savefig(image_path)
    plt.close()

    return image_path


def task_2(dataUrl: str) -> str:
    dataset = ld.loadDataset(
        dataUrl, "temp.csv", colNames=["timestamp", "temperature"], skip=10)

    dataset["timestamp"] = pd.to_datetime(
        dataset["timestamp"], format="%Y%m%dT%H%M")
    dataset["date"] = dataset["timestamp"].dt.date

    daily_min = dataset.groupby("date")["temperature"].min().reset_index()
    daily_min["year"] = pd.to_datetime(daily_min["date"]).dt.year
    daily_min["week"] = pd.to_datetime(daily_min["date"]).dt.strftime("%U")

    weekly_min = daily_min[daily_min["week"] == "01"]

    weekly_pivot = weekly_min.pivot(
        index="date",
        columns="year",
        values="temperature")
    weekly_pivot.index = pd.to_datetime(weekly_pivot.index).strftime("%a")

    plt.figure(figsize=(10, 5))
    markers = ['o', 's', 'D', 'v', '^']

    for i, year in enumerate(weekly_pivot.columns):
        plt.plot(weekly_pivot.index,
                 weekly_pivot[year],
                 marker=markers[i % len(markers)],
                 linestyle='-',
                 label=str(year))

    plt.xlabel("Day of the Week")
    plt.ylabel("Minimum Temperature (°C)")
    plt.title("Weekly Minimum Temperature Over the Years in Basel")
    plt.legend()
    plt.grid(True)

    output_path = "task_2_chart.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def task_3(dataUrl: str) -> None:
    sns.set_style("whitegrid")
    dataset = ld.loadDataset(dataUrl)

    print(dataset)

    scatter = sns.lmplot(
        data=dataset, x="bmi", y="charges", hue="smoker",
        height=7, aspect=1.6, palette={"yes": "red", "no": "blue"},
        scatter_kws=dict(s=60, linewidths=.7, edgecolors='black')
    )
    scatter.ax

    plt.title(
        "Залежність між ІМТ та медичними витратами з урахуванням статусу курця")
    plt.xlabel("Індекс маси тіла (BMI)")
    plt.ylabel("Медичні витрати (USD)")
    plt.legend(title="Куріння")

    plt.show()


def task_3_2(dataUrl: str) -> None:
    dataset = ld.loadDataset(
        dataUrl, columnsToRender=[
            "radius_mean", "radius_worst"])
    print(dataset)
    x = dataset['radius_mean']
    y = dataset['radius_worst']

    ccs = stattools.ccf(x, y)[:100]
    nlags = len(ccs)

    conf_level = 2 / np.sqrt(nlags)

    plt.figure(figsize=(12, 7), dpi=80)

    plt.hlines(0, xmin=0, xmax=100, color='gray')
    plt.hlines(conf_level, xmin=0, xmax=100, color='gray')
    plt.hlines(-conf_level, xmin=0, xmax=100, color='gray')

    plt.bar(x=np.arange(len(ccs)), height=ccs, width=.3)

    plt.title(
        '$Cross\\; Correlation\\; Plot:\\; radius mean \\; vs\\; radius worst$',
        fontsize=22)
    plt.xlim(0, len(ccs))
    plt.show()


def task_3_3(dataUrl: str) -> None:
    dataset = ld.loadDataset(
        dataUrl, "temp2.csv", colNames=[
            "date", "rerender1", "rerender2", "meanTemp",], columnsToRender=[
            "date", "meanTemp"], skip=10)
    print(dataset)

    dataset["date"] = pd.to_datetime(dataset["date"], format='%Y%m%dT%H%M')

    print(dataset)
    dataset.set_index('date', inplace=True)

    plt.figure(figsize=(16, 10), dpi=80)
    calmap.calendarplot(
        dataset['meanTemp'], fig_kws={
            'figsize': (
                16, 10)}, yearlabel_kws={
            'color': 'black', 'fontsize': 14})
    plt.show()


def task_4(dataUrl: str) -> None:
    dataset = ld.loadDataset(dataUrl)
    print(dataset)
    df_counts = dataset.groupby(
        ['Year', 'Category']).size().reset_index(name='Count')

    df_yearly_counts = dataset.groupby(
        'Year').size().reset_index(name='Total_Laureates')

    df_organizations = dataset[dataset['Organization Name'].notna()]
    org_counts = df_organizations['Organization Name'].value_counts().head(10)

    dataset['Birth Date'] = pd.to_datetime(
        dataset['Birth Date'], errors='coerce')
    dataset['Death Date'] = pd.to_datetime(
        dataset['Death Date'], errors='coerce')
    dataset['Age'] = dataset.apply(
        lambda row: calculate_age(
            row['Birth Date'],
            row['Death Date']),
        axis=1)

    plt.figure(figsize=(15, 12))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    categories = df_counts['Category'].unique()

    for i, category in enumerate(categories):
        ax = axes[i]
        category_data = df_counts[df_counts['Category'] == category]
        sns.lineplot(
            data=category_data,
            x='Year',
            y='Count',
            marker='o',
            ax=ax,
            color='b')
        ax.set_title(f"Категорія: {category}")
        ax.set_xlabel("Рік")
        ax.set_ylabel("Кількість лауреатів")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    sns.lineplot(
        data=df_yearly_counts,
        x='Year',
        y='Total_Laureates',
        marker='o',
        ax=axes2[0],
        color='b')
    axes2[0].set_title("Тренд кількості лауреатів за рік")
    axes2[0].set_xlabel("Рік")
    axes2[0].set_ylabel("Загальна кількість лауреатів")

    sns.barplot(
        x=org_counts.values,
        y=org_counts.index,
        palette="viridis",
        ax=axes2[1])
    axes2[1].set_title("Топ організацій, які найчастіше перемагали")
    axes2[1].set_xlabel("Кількість перемог")
    axes2[1].set_ylabel("Організація")

    sns.boxplot(
        data=dataset,
        x='Category',
        y='Age',
        palette="Set2",
        ax=axes2[2])
    axes2[2].set_title("Віковий розподіл переможців у різних категоріях")
    axes2[2].set_xlabel("Категорія премії")
    axes2[2].set_ylabel("Вік лауреата")
    axes2[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    plt.show()


def calculate_age(birth_date, death_date):
    if pd.isna(birth_date):
        return None
    if pd.isna(death_date):
        death_date = pd.Timestamp.today()
    return death_date.year - birth_date.year - \
        ((death_date.month, death_date.day) < (birth_date.month, birth_date.day))


def task_5():
    dataset = gapminder.copy()

    scatter_fig = px.scatter(
        dataset,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        title="GDP vs Life Expectancy")

    countries = ["India", "China", "United States", "Japan"]
    df_selected = dataset[dataset["country"].isin(countries)]

    line_fig = px.line(
        df_selected,
        x="year",
        y="pop",
        color="country",
        markers=True,
        title="Population Over Years for Selected Countries")

    scatter_fig.show()
    line_fig.show()


def task_indz_1(dataUrl: str):
    dataset = ld.loadDatasetLocal(dataUrl)

    print("\n--- Перші рядки ---")
    print(dataset.head())

    print("\n--- Розмірність ---")
    print(dataset.shape)

    print("\n--- Типи даних ---")
    print(dataset.dtypes)

    print("\n--- Пропущені значення ---")
    print(dataset.isnull().sum())

    print("\n--- Статистичний опис ---")
    print(dataset.describe())

    print("\n--- Розподіл 'Survived' (%) ---")
    print(dataset['Survived'].value_counts(normalize=True) * 100)

    for col in ['Sex', 'Pclass', 'Embarked']:
        if col in dataset.columns:
            print(f"\n--- Розподіл {col} ---")
            print(dataset[col].value_counts())

    missing = dataset.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\n--- Детальний аналіз пропущених значень ---")
        print(missing.sort_values(ascending=False))

    numeric_cols = dataset.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        sns.histplot(dataset[col].dropna(), kde=True)
        plt.title(f"Розподіл {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.show()

    for col in numeric_cols:
        if col != 'Survived':
            sns.boxplot(x='Survived', y=col, data=dataset)
            plt.title(f"{col} за класами 'Survived'")
            plt.xlabel("Вижив")
            plt.ylabel(col)
            plt.show()

    for col in ['Age', 'Fare']:
        if col in dataset.columns:
            sns.violinplot(x='Survived', y=col, data=dataset)
            plt.title(f"Розподіл {col} залежно від виживання")
            plt.xlabel("Вижив")
            plt.ylabel(col)
            plt.show()

    for col in ['Sex', 'Pclass', 'Embarked']:
        if col in dataset.columns:
            survived_by_cat = pd.crosstab(dataset[col], dataset['Survived'], normalize='index') * 100
            survived_by_cat.plot(kind='bar', stacked=True)
            plt.title(f"Виживання залежно від {col}")
            plt.ylabel("Відсоток")
            plt.xlabel(col)
            plt.legend(title='Вижив')
            plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Матриця кореляції")
    plt.show()


def task_indz_2(dataUrl: str):
    dataset = ld.loadDatasetLocal(dataUrl)

    print("\n--- Перші рядки ---")
    print(dataset.head())

    print("\n--- Розмірність ---")
    print(dataset.shape)

    print("\n--- Типи даних ---")
    print(dataset.dtypes)

    print("\n--- Пропущені значення ---")
    print(dataset.isnull().sum())

    print("\n--- Статистичний опис ---")
    print(dataset.describe())

    print("\n--- Розподіл цільової змінної ---")
    print(dataset['target'].value_counts())

    numeric_cols = dataset.select_dtypes(include='number').columns.drop('target', errors='ignore')
    for col in numeric_cols:
        sns.histplot(dataset[col], kde=True)
        plt.title(f"Розподіл {col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.show()

        sns.boxplot(x='target', y=col, data=dataset)
        plt.title(f"{col} за класами target")
        plt.xlabel("Клас")
        plt.ylabel(col)
        plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Матриця кореляції")
    plt.show()

    if len(numeric_cols) <= 6:
        sns.pairplot(dataset, hue='target')
        plt.suptitle("Парні графіки ознак", y=1.02)
        plt.show()

    cluster_and_visualize(dataset)


def cluster_and_visualize(dataset: pd.DataFrame, n_clusters: int = 3):
    if 'target' not in dataset.columns:
        print("Цільова змінна 'target' не знайдена у наборі даних.")
        return

    X = dataset.drop(columns=['target'])
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, cluster_labels)
    ari_score = adjusted_rand_score(dataset['target'], cluster_labels)

    print(f"\nКоефіцієнт силуету: {sil_score:.4f}")
    print(f"Індекс скоригованої випадковості (ARI): {ari_score:.4f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                    hue=cluster_labels, palette='viridis', s=60)
    plt.title("Кластери після PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title='Кластери')
    plt.grid(True)
    plt.show()