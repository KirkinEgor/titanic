import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Titanic Dashboard")
st.title("Анализ пассажиров Титаника")
st.markdown("Дашборд для исследовательского анализа данных о пассажирах Титаника")


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Titanic-Dataset.csv")
    except FileNotFoundError:
        uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл Titanic", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Файл Titanic-Dataset.csv не найден. Пожалуйста, загрузите его.")
            st.stop()
    return df

df = load_data()


st.header("📊 Описательная статистика")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Размер датасета")
    st.write(f"**Строк:** {df.shape[0]}")
    st.write(f"**Столбцов:** {df.shape[1]}")

with col2:
    st.subheader("Типы данных")
    st.dataframe(pd.DataFrame(df.dtypes).rename(columns={0: "Тип данных"}))


st.subheader("Пропуски в данных")
missing = pd.DataFrame({
    "Колонка": df.columns,
    "Количество пропусков": df.isnull().sum(),
    "Доля (%)": (df.isnull().sum() / len(df) * 100).round(2)
})
st.dataframe(missing[missing["Количество пропусков"] > 0])


st.header("📄 Просмотр данных")
n_rows = st.slider("Количество строк для отображения", min_value=5, max_value=50, value=10, step=5)
st.dataframe(df.head(n_rows))


st.header("📈 Визуализация")

# 1. Распределение выживших (bar plot с подписями)
st.subheader("1. Выживание пассажиров")
survived_counts = df['Survived'].value_counts().sort_index()
survived_labels = {0: 'Не выжил', 1: 'Выжил'}
fig1 = px.bar(x=[survived_labels[i] for i in survived_counts.index],
              y=survived_counts.values,
              labels={'x': 'Статус', 'y': 'Количество'},
              title='Количество выживших и погибших',
              text=survived_counts.values,
              color=[survived_labels[i] for i in survived_counts.index],
              color_discrete_sequence=['#EF553B', '#636EFA'])
fig1.update_traces(textposition='outside', marker_line_width=1, marker_line_color='black')
fig1.update_layout(showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# 2. Распределение по классам билета (дискретные цвета)
st.subheader("2. Классы пассажиров")
class_counts = df['Pclass'].value_counts().sort_index()
fig2 = px.bar(x=class_counts.index, y=class_counts.values,
              labels={'x': 'Класс', 'y': 'Количество пассажиров'},
              title='Распределение по классам обслуживания',
              text=class_counts.values,
              color=class_counts.index.astype(str),
              color_discrete_map={'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c'})
fig2.update_traces(textposition='outside', marker_line_width=1, marker_line_color='black')
fig2.update_layout(showlegend=False)
fig2.update_xaxes(type='category')
st.plotly_chart(fig2, use_container_width=True)

# 3. Возрастное распределение
st.subheader("3. Возрастное распределение")
fig3 = px.histogram(df, x="Age", nbins=30, 
                    labels={"Age": "Возраст", "count": "Количество"},
                    title="Гистограмма возраста пассажиров",
                    color_discrete_sequence=["#00CC96"])
st.plotly_chart(fig3, use_container_width=True)

# 4. Зависимость выживания от пола (круговая диаграмма)
st.subheader("4. Выживаемость пассажиров по полу")
survival_by_sex = df.groupby(["Sex", "Survived"]).size().reset_index(name="count")
fig4 = px.pie(survival_by_sex, values="count", names="Sex", facet_col="Survived",
              title="Выживание по полу", hole=0.3,
              color="Sex", color_discrete_sequence=["#FFA15A", "#B6E880"])
st.plotly_chart(fig4, use_container_width=True)

# 5. Распределение возраста по классам (boxplot)
st.subheader("5. Возраст в разрезе классов")
fig5 = px.box(df, x="Pclass", y="Age", color="Pclass",
              labels={"Pclass": "Класс", "Age": "Возраст"},
              title="Распределение возраста по классам",
              color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig5, use_container_width=True)


st.header("Интерактивная гистограмма")
st.markdown("Выберите переменную для построения гистограммы. Распределение показано отдельно для выживших и погибших.")


numeric_cols_for_hist = df.select_dtypes(include=[np.number]).columns.tolist()
if "PassengerId" in numeric_cols_for_hist:
    numeric_cols_for_hist.remove("PassengerId")
if "Survived" in numeric_cols_for_hist:
    numeric_cols_for_hist.remove("Survived")


categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
exclude_cols = ["Name", "Ticket", "Cabin"]
categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
# Pclass - числовой, но может быть категориальным, добавим вручную
if "Pclass" not in numeric_cols_for_hist and "Pclass" in df.columns:
    categorical_cols.append("Pclass")

available_cols = numeric_cols_for_hist + categorical_cols

selected_var = st.selectbox("Выберите признак для гистограммы", available_cols)

bins = None
if selected_var in numeric_cols_for_hist:
    bins = st.slider("Количество интервалов (бинов)", min_value=5, max_value=50, value=20, step=5)

if selected_var in numeric_cols_for_hist:
    fig_hist = px.histogram(df, x=selected_var, color="Survived", 
                            nbins=bins,
                            labels={"Survived": "Выживание", selected_var: selected_var},
                            title=f"Распределение '{selected_var}' в зависимости от выживания",
                            color_discrete_map={0: "#EF553B", 1: "#636EFA"},
                            barmode="group")  # <-- ключевое изменение: группировка вместо наложения
else:
    grouped = df.groupby([selected_var, "Survived"]).size().reset_index(name="count")
    fig_hist = px.bar(grouped, x=selected_var, y="count", color="Survived",
                      labels={"Survived": "Выживание", "count": "Количество"},
                      title=f"Распределение '{selected_var}' в зависимости от выживания",
                      color_discrete_map={0: "#EF553B", 1: "#636EFA"},
                      barmode="group")

fig_hist.update_layout(bargap=0.1)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("Дашборд создан на основе датасета Titanic. Данные содержат информацию о пассажирах, включая выживаемость, класс, пол, возраст и другие атрибуты.")