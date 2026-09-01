import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from plotly.subplots import make_subplots
import pydeck as pdk


st.title("TP2 - Dados da COVID-19 no Brasil")

# Carregamento de arquivos
df = pd.concat([pd.read_csv("Entrega TP2/Dados/HIST_PAINEL_COVIDBR_05set2025 (1)/HIST_PAINEL_COVIDBR_2020_Parte1_05set2025.csv", sep=";", encoding="utf-8", low_memory=False), pd.read_csv("Entrega TP2/Dados/HIST_PAINEL_COVIDBR_05set2025 (1)/HIST_PAINEL_COVIDBR_2020_Parte2_05set2025.csv", sep=";", encoding="utf-8", low_memory=False)], ignore_index=True)

# conversão para organizar as datas nos gráficos
df["data"] = pd.to_datetime(df["data"], errors="coerce")

#separação da base por estado
df_estado = df[(df["estado"].notna()) & (df["municipio"].isna())]
df_estado_semana = df_estado.groupby(["regiao", "estado", "semanaEpi"]).agg({"data": "min", "casosAcumulado": "max", "casosNovos": "sum", "obitosNovos": "sum"}).reset_index()
df_regiao_semana = df_estado_semana.groupby(["regiao", "semanaEpi"]).agg({"data": "min", "casosNovos": "sum", "obitosNovos": "sum"}).reset_index()

st.write("Previa dos DadosS")
st.dataframe(df.head(10), width="stretch")


# Exercício 1
st.subheader("-- Exercício 1 --")
st.header("Importância da visualização de dados: ")
st.write("A visualização de dados é muito importante pois facilita a compreensão de uma grande quantidade de informações. Por meio de gráficos e mapas é possível acompanhar o aumento ou a diminuição dos casos de covid e óbitos, comparar estados e regiões e identificar os locais que foram mais afetados e precisam de mais atenção.")
st.write("Essas informações ajudam a população a entender a situação da pandemia e a importância das medidas de prevenção.")


# Exercício 2
st.subheader("-- Exercício 2 --")
st.header("Gráfico de barras com Streamlit")
maior_semana = int(df_estado_semana[df_estado_semana["estado"] == "SP"].nlargest(1, "casosNovos")["semanaEpi"].iloc[0])
maximo_casos_semana = int(df_estado_semana[df_estado_semana["estado"] == "SP"]["casosNovos"].max())
st.bar_chart(df_estado_semana[df_estado_semana["estado"] == "SP"], x="semanaEpi", y="casosNovos")
st.write(f"O estado escolhido foi **São Paulo** por possuir uma grande "
         "população e uma região metropolitana com intensa circulação de pessoas. "
         f"Em 2020, a maior quantidade de casos novos aparece na semana {maior_semana}, com {maximo_casos_semana} casos nesta semana.")


# Exercício 3
st.subheader("-- Exercício 3 --")
st.header("Gráfico de linha com Streamlit")

df_linha = df[(df["regiao"] == "Brasil") & (df["estado"].isna()) & (df["municipio"].isna())].groupby("semanaEpi").agg({"data": "min", "obitosAcumulado": "max"}).reset_index().sort_values("data")
df_linha["aumentoSemanal"] = df_linha["obitosAcumulado"].diff()
obitos_inicio = int(df_linha["obitosAcumulado"].iloc[0])
obitos_final = int(df_linha["obitosAcumulado"].iloc[-1])
maior_aumento = df_linha.loc[df_linha["aumentoSemanal"].idxmax()]

st.line_chart(df_linha, x="data", y="obitosAcumulado")
st.write("Em **2020**, os óbitos iniciaram em Abril subindo muito rápido chegando a quase 200.000 no final de **2020**")


# Exercício 4
st.subheader("-- Exercício 4 --")
st.header("Gráfico de área com Streamlit")

df_area = df_estado_semana[df_estado_semana["estado"].isin(["RJ", "SP", "MG"])].pivot(index="data", columns="estado", values="casosAcumulado")

st.area_chart(df_area)
st.write("Foram escolhidos Rio de Janeiro, São Paulo e Minas Gerais, que são estados da região Sudeste. "
         "São Paulo nitidamente apresenta os maiores valores acumulados. "
         "O principal motivo pode ser porque possui a maior população do que os outros três")
#df_coordenadas para o exercicio 5 e 12
#pesquisei as coordenadas no Google Maps
#pesquisei a área dos municípios no IBGE
df_coordenadas = pd.DataFrame({"estado": ["RJ", "RJ", "RJ", "RJ", "RJ", "SP", "SP", "SP", "SP", "SP", "MG", "MG", "ES", "ES"],
                               "municipio": ["Rio de Janeiro", "Niterói", "São Gonçalo", "Belford Roxo", "Duque de Caxias", "São Paulo", "Campinas", "São José do Rio Preto", "São Bernardo do Campo", "Santos", "Belo Horizonte", "Uberlândia", "Vitória", "Vila Velha"],
                               "latitude": [-22.9068, -22.8832, -22.8269, -22.7640, -22.7856, -23.5505, -22.9099, -20.8197, -23.6914, -23.9342, -19.9167, -18.9186, -20.3155, -20.3297],
                               "longitude": [-43.1729, -43.1034, -43.0539, -43.3992, -43.3117, -46.6333, -47.0626, -49.3794, -46.5646, -46.3286, -43.9345, -48.2772, -40.3128, -40.2925],
                               "areaKm2": [1200.329, 133.757, 248.160, 78.985, 467.319, 1521.202, 794.571, 431.944, 409.532, 281.033, 331.354, 4115.206, 97.123, 210.225]})

df_coordenadas = df_coordenadas.merge(df[df["municipio"].notna()].sort_values("data").groupby(["estado", "municipio"], as_index=False).tail(1)[["estado", "municipio", "casosAcumulado", "populacaoTCU2019"]], on=["estado", "municipio"], how="left")

# Exercício 5
st.subheader("-- Exercício 5 --")
st.header("Mapa com Streamlit")

df_mapa_sao_paulo = df_coordenadas[df_coordenadas["estado"] == "SP"].sort_values("casosAcumulado", ascending=False).head(5)

st.map(df_mapa_sao_paulo, latitude="latitude", longitude="longitude", size=20000, color="#1f77b4aa", height=450)
st.write(df_mapa_sao_paulo[["municipio", "casosAcumulado", "latitude", "longitude"]])
st.write("O estado escolhido foi **São Paulo**. "
         "Os cinco municípios com maior número de casos acumulados são São Paulo, Campinas, São José do Rio Preto, São Bernardo do Campo e Santos. "
         "O município de São Paulo aparece em primeiro lugar, maior quantidade de casos acumulados.")


# Exercício 6
st.subheader("-- Exercício 6 --")
st.header("Visualização com Matplotlib")

df_matplotlib = df_estado_semana[df_estado_semana["semanaEpi"] == df_estado_semana["semanaEpi"].max()].sort_values("estado")
fig, (ax_casos, ax_obitos) = plt.subplots(1, 2, figsize=(16, 6))

ax_casos.bar(df_matplotlib["estado"], df_matplotlib["casosNovos"], color="tab:blue")
ax_casos.set_title("Casos novos por estado")
ax_casos.set_xlabel("Estado")
ax_casos.set_ylabel("Casos novos")
ax_casos.tick_params(axis="x", rotation=90)
ax_casos.grid(axis="y", alpha=0.3)

ax_obitos.bar(df_matplotlib["estado"], df_matplotlib["obitosNovos"], color="tab:orange")
ax_obitos.set_title("Óbito por estado")
ax_obitos.set_xlabel("Estado")
ax_obitos.set_ylabel("Óbitos")
ax_obitos.tick_params(axis="x", rotation=90)
ax_obitos.grid(axis="y", alpha=0.3)

fig.suptitle("Maior semana de 2020")
fig.tight_layout()

st.pyplot(fig)
st.write("Os estados com mais casos novos apresentam mais óbitos.")


# Exercício 7
st.subheader("-- Exercício 7 --")
st.header("Boxplot com Seaborn")

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_regiao_semana[df_regiao_semana["regiao"].isin(["Norte", "Nordeste", "Sudeste"])], x="regiao", y="casosNovos", ax=ax)
ax.set_title("Distribuição dos casos novos por semana")
ax.set_xlabel("Região")
ax.set_ylabel("Casos novos")

st.pyplot(fig)
st.write("O boxplot permite comparar a distribuição dos casos nas três regiões. "
         "A região Sudeste apresenta valores maiores, porém se trata da região mais populosa entre as três.")


# Exercício 8
st.subheader("-- Exercício 8 --")
st.header("Gráfico de área com Altair")

grafico_area = alt.Chart(df_regiao_semana[df_regiao_semana["regiao"] == "Sudeste"]).mark_area().encode(x=alt.X("data:T", title="Data"), y=alt.Y("casosNovos:Q", title="Casos novos"))

st.altair_chart(grafico_area, width="stretch")
st.write("A região escolhida foi o **Sudeste** por ser a região mais populosa do Brasil. "
         "O gráfico permite visualizar os períodos de aumento e diminuição dos casos novos.")


# Exercício 9
st.subheader("-- Exercício 9 --")
st.header("Heatmap com Altair")

df_correlacao = df_estado[df_estado["estado"] == "SP"][["casosNovos", "obitosNovos"]].corr()
valor_correlacao = round(df_correlacao.loc["casosNovos", "obitosNovos"], 2)
df_correlacao = df_correlacao.reset_index().melt(id_vars="index")
heatmap = alt.Chart(df_correlacao).mark_rect().encode(x=alt.X("index:N", title="Indicador"), y=alt.Y("variable:N", title="Indicador"), color=alt.Color("value:Q", title="Correlação"), tooltip=["index", "variable", "value"])

st.altair_chart(heatmap, width="stretch")
st.write(f"No estado de **São Paulo**, o heatmap mostra uma correlação de **{valor_correlacao}** entre os casos novos e os óbitos novos. O que indica uma alta correlação")


# Exercício 10
st.subheader("-- Exercício 10--")
st.header("Gráfico de pizza com Plotly")

df_pizza = df_estado.sort_values("data").groupby("estado", as_index=False).tail(1).groupby("regiao")["casosAcumulado"].sum().reset_index()
grafico_pizza = px.pie(df_pizza, names="regiao", values="casosAcumulado", title="Distribuição dos casos acumulados entre as regiões")

st.plotly_chart(grafico_pizza, width="stretch")
st.write("O gráfico mostra a participação de cada região no total de casos. Sudeste apresenta a maior fatia, o que é esperado devido a alta população")


# Exercício 11
st.subheader("-- Exercício 11 --")
st.header("Subplots com Plotly")

grafico_sudeste = px.bar(df_regiao_semana[df_regiao_semana["regiao"] == "Sudeste"], x="semanaEpi", y=["casosNovos", "obitosNovos"], barmode="group", title="Sudeste")
grafico_nordeste = px.bar(df_regiao_semana[df_regiao_semana["regiao"] == "Nordeste"], x="semanaEpi", y=["casosNovos", "obitosNovos"], barmode="group", title="Nordeste")
grafico_subplots = make_subplots(rows=1, cols=2, subplot_titles=["Sudeste", "Nordeste"])

for trace in grafico_sudeste.data:
    grafico_subplots.add_trace(trace, row=1, col=1)

for trace in grafico_nordeste.data:
    grafico_subplots.add_trace(trace, row=1, col=2)

grafico_subplots.update_layout(title="Casos e óbitos novos por semana em 2020", barmode="group")

st.plotly_chart(grafico_subplots, width="stretch")
st.write("O Sudeste possui valores maiores em várias semanas, enquanto o Nordeste apresenta uma quantidade consideravemente menor.")


# Exercício 12
st.subheader("-- Exercício 12 --")
st.header("Mapa interativo com Pydeck")

df_pydeck = df_coordenadas.dropna(subset=["casosAcumulado", "populacaoTCU2019", "areaKm2"]).copy()
df_pydeck["densidadePopulacional"] = (df_pydeck["populacaoTCU2019"] / df_pydeck["areaKm2"]).round(2) #densidade populacional do município, usada na cor da coluna
df_pydeck["casosPor100MilHabitantes"] = (df_pydeck["casosAcumulado"] / df_pydeck["populacaoTCU2019"] * 100000).round(2) #casos por 100 mil habitantes, altura da coluna
df_pydeck["cor"] = df_pydeck["densidadePopulacional"].apply(lambda densidade: [int(50 + densidade / df_pydeck["densidadePopulacional"].max() * 205), 60, 100, 200])
mapa_pydeck = pdk.Deck(initial_view_state=pdk.ViewState(latitude=-22.0, longitude=-45.0, zoom=5, pitch=50), layers=[pdk.Layer("ColumnLayer", data=df_pydeck, get_position=["longitude", "latitude"], get_elevation="casosPor100MilHabitantes", elevation_scale=30, radius=15000, get_fill_color="cor", pickable=True)], tooltip={"text": "{municipio} - {estado}\nCasos acumulados: {casosAcumulado}\nCasos por 100 mil habitantes: {casosPor100MilHabitantes}\nDensidade populacional: {densidadePopulacional} hab/km²"})

st.pydeck_chart(mapa_pydeck)
st.write(df_pydeck[["municipio", "estado", "casosAcumulado", "casosPor100MilHabitantes", "densidadePopulacional"]])
st.write("O mapa apresenta municípios da região **Sudeste**. "
         "A altura das colunas mostra os casos acumulados por 100 mil habitantes e a cor representa a densidade populacional. "
         "As colunas mais avermelhadas indicam municípios com mais habitantes por km². "
         "Em municípios mais densos existe maior proximidade entre as pessoas, o que pode facilitar a disseminação da COVID-19.")
