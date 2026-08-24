"""TP2: visualizacoes da COVID-19 com dados do Ministerio da Saude."""

from __future__ import annotations

import io
import unicodedata
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="TP2 - COVID-19", page_icon="📊", layout="wide")

REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
REGIAO_POR_UF = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste", "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste",
    "SP": "Sudeste", "PR": "Sul", "RS": "Sul", "SC": "Sul",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
}
UF_POR_CODIGO = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
EXERCICIOS = [
    "1. Importancia da visualizacao", "2. Barras com Streamlit",
    "3. Linha com Streamlit", "4. Area com Streamlit", "5. Mapa com Streamlit",
    "6. Barras com Matplotlib", "7. Boxplot com Seaborn", "8. Area com Altair",
    "9. Heatmap com Altair", "10. Pizza com Plotly", "11. Subplots com Plotly",
    "12. Mapa interativo com PyDeck",
]

# Coordenadas de apoio caso a fonte on-line de todos os municipios esteja indisponivel.
COORDENADAS_RJ = [
    ("Rio de Janeiro", -22.9068, -43.1729), ("Niteroi", -22.8832, -43.1034),
    ("Sao Goncalo", -22.8269, -43.0539), ("Duque de Caxias", -22.7856, -43.3117),
    ("Nova Iguacu", -22.7556, -43.4603), ("Campos dos Goytacazes", -21.7622, -41.3181),
    ("Petropolis", -22.5112, -43.1779), ("Volta Redonda", -22.5202, -44.0996),
    ("Macae", -22.3768, -41.7848), ("Belford Roxo", -22.7640, -43.3990),
    ("Itaborai", -22.7441, -42.8597), ("Marica", -22.9195, -42.8186),
    ("Cabo Frio", -22.8894, -42.0286), ("Nova Friburgo", -22.2819, -42.5311),
    ("Angra dos Reis", -23.0067, -44.3181), ("Barra Mansa", -22.5447, -44.1713),
    ("Resende", -22.4703, -44.4509), ("Teresopolis", -22.4167, -42.9782),
    ("Sao Joao de Meriti", -22.8058, -43.3729), ("Nilopolis", -22.8057, -43.4233),
]


def chave_texto(valor: object) -> str:
    """Remove acentos e sinais para comparar nomes vindos de fontes diferentes."""
    texto = "" if pd.isna(valor) else str(valor)
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "".join(c for c in texto.lower() if c.isalnum())


ALIASES = {
    "regiao": "regiao", "estado": "estado", "municipio": "municipio",
    "data": "data", "semanaepi": "semana_epi", "populacaotcu2019": "populacao",
    "populacao": "populacao", "casosacumulado": "casos_acumulados",
    "casosacumulados": "casos_acumulados", "casosnovos": "casos_novos",
    "obitosacumulado": "obitos_acumulados", "obitosacumulados": "obitos_acumulados",
    "obitosnovos": "obitos_novos", "codmun": "codigo_municipio",
    "codigomunicipio": "codigo_municipio", "leitosocupados": "leitos_ocupados",
    "ocupacaodeleitos": "leitos_ocupados",
}


def normalizar(chunk: pd.DataFrame) -> pd.DataFrame:
    """Padroniza as colunas das diferentes partes do historico oficial."""
    renomear = {}
    for coluna in chunk.columns:
        chave = chave_texto(coluna)
        if chave in ALIASES:
            renomear[coluna] = ALIASES[chave]
        elif "leito" in chave and "ocup" in chave:
            renomear[coluna] = "leitos_ocupados"
    chunk = chunk.rename(columns=renomear)
    essenciais = [
        "regiao", "estado", "municipio", "data", "semana_epi", "populacao",
        "casos_acumulados", "casos_novos", "obitos_acumulados", "obitos_novos",
        "codigo_municipio",
    ]
    for coluna in essenciais:
        if coluna not in chunk:
            chunk[coluna] = pd.NA
    manter = essenciais + (["leitos_ocupados"] if "leitos_ocupados" in chunk else [])
    chunk = chunk[manter].copy()
    for coluna in ["regiao", "estado", "municipio"]:
        chunk[coluna] = chunk[coluna].astype("string").str.strip()
        chunk[coluna] = chunk[coluna].replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    chunk["data"] = pd.to_datetime(chunk["data"], errors="coerce", dayfirst=True)
    numericas = [
        "semana_epi", "populacao", "casos_acumulados", "casos_novos",
        "obitos_acumulados", "obitos_novos", "codigo_municipio",
    ] + (["leitos_ocupados"] if "leitos_ocupados" in chunk else [])
    for coluna in numericas:
        chunk[coluna] = pd.to_numeric(chunk[coluna], errors="coerce")
    chunk["regiao"] = chunk["regiao"].fillna(chunk["estado"].map(REGIAO_POR_UF))
    chunk["ano"] = chunk["data"].dt.year.astype("Int64")
    chunk["semana_epi"] = chunk["semana_epi"].astype("Int64")
    chunk["ordem_semana"] = chunk["ano"] * 100 + chunk["semana_epi"]
    chunk["periodo_semana"] = (
        chunk["ano"].astype("string") + "-SE"
        + chunk["semana_epi"].astype("string").str.zfill(2)
    )
    return chunk


def consumir_csv(arquivo, nome, superiores, municipais):
    """Le em blocos para nao manter todo o historico municipal na memoria."""
    amostra = arquivo.read(8192)
    arquivo.seek(0)
    texto = amostra.decode("utf-8", errors="ignore") if isinstance(amostra, bytes) else amostra
    separador = ";" if texto.count(";") >= texto.count(",") else ","
    try:
        leitor = pd.read_csv(
            arquivo, sep=separador, encoding="utf-8", encoding_errors="replace",
            chunksize=150_000, low_memory=False,
        )
        for chunk in leitor:
            chunk = normalizar(chunk)
            if chunk["data"].notna().sum() == 0:
                continue
            mascara = chunk["municipio"].notna()
            superiores.append(chunk.loc[~mascara])
            municipal = chunk.loc[mascara]
            if not municipal.empty:
                municipais.append(
                    municipal.sort_values("data")
                    .groupby(["estado", "municipio"], dropna=False, as_index=False).tail(1)
                )
    except Exception as erro:
        raise ValueError(f"Nao foi possivel ler {nome}: {erro}") from erro


@st.cache_data(show_spinner=False, max_entries=2)
def carregar_dados(payloads: tuple[tuple[str, bytes], ...]) -> pd.DataFrame:
    superiores, municipais = [], []
    for nome, conteudo in payloads:
        if nome.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(conteudo)) as pacote:
                membros = [m for m in pacote.namelist() if m.lower().endswith(".csv")]
                if not membros:
                    raise ValueError(f"{nome} nao contem CSVs.")
                for membro in membros:
                    with pacote.open(membro) as csv_zip:
                        consumir_csv(csv_zip, membro, superiores, municipais)
        else:
            consumir_csv(io.BytesIO(conteudo), nome, superiores, municipais)
    if not superiores and not municipais:
        raise ValueError("Nenhuma linha valida foi encontrada.")
    partes = []
    if superiores:
        partes.append(pd.concat(superiores, ignore_index=True))
    if municipais:
        municipal = pd.concat(municipais, ignore_index=True)
        municipal = municipal.sort_values("data").groupby(
            ["estado", "municipio"], dropna=False, as_index=False
        ).tail(1)
        partes.append(municipal)
    return pd.concat(partes, ignore_index=True)


@st.cache_data(ttl=86_400, show_spinner=False)
def carregar_coordenadas() -> tuple[pd.DataFrame, bool]:
    """Usa coordenadas publicas como informacao geografica auxiliar."""
    url = "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv"
    try:
        coord = pd.read_csv(url)
        coord["estado"] = coord["codigo_uf"].map(UF_POR_CODIGO)
        coord["municipio_chave"] = coord["nome"].map(chave_texto)
        return coord[["estado", "municipio_chave", "latitude", "longitude"]], True
    except Exception:
        coord = pd.DataFrame(COORDENADAS_RJ, columns=["nome", "latitude", "longitude"])
        coord["estado"] = "RJ"
        coord["municipio_chave"] = coord["nome"].map(chave_texto)
        return coord[["estado", "municipio_chave", "latitude", "longitude"]], False


def linhas_estaduais(dados):
    return dados.loc[dados["estado"].notna() & dados["municipio"].isna()].copy()


def linhas_regionais(dados):
    return dados.loc[
        dados["estado"].isna() & dados["municipio"].isna()
        & dados["regiao"].isin(REGIOES)
    ].copy()


def linhas_municipais(dados):
    return dados.loc[dados["estado"].notna() & dados["municipio"].notna()].copy()


def semanal_novos(base, grupos):
    colunas = grupos + ["ordem_semana", "periodo_semana"]
    return (
        base.dropna(subset=["ordem_semana"]).groupby(colunas, as_index=False, dropna=False)
        .agg(data_inicio=("data", "min"), casos_novos=("casos_novos", "sum"),
             obitos_novos=("obitos_novos", "sum"))
        .sort_values("ordem_semana")
    )


def semanal_regioes(dados):
    base = linhas_regionais(dados)
    if base.empty:
        base = linhas_estaduais(dados)
    return semanal_novos(base, ["regiao"])


def selecionar(rotulo, opcoes, padrao, chave):
    indice = opcoes.index(padrao) if padrao in opcoes else 0
    return st.selectbox(rotulo, opcoes, index=indice, key=chave)


st.title("TP2 - Visualizacoes da COVID-19 no Brasil")
st.caption(
    "Fonte epidemiologica: Painel Coronavirus Brasil, Ministerio da Saude. "
    "As coordenadas sao apenas apoio geografico."
)

with st.sidebar:
    st.header("Dados e navegacao")
    st.markdown(
        "1. Acesse [covid.saude.gov.br](https://covid.saude.gov.br/).  \n"
        "2. Clique em **Arquivo CSV**.  \n3. Envie abaixo o ZIP ou os CSVs extraidos."
    )
    arquivos = st.file_uploader(
        "Arquivo(s) oficial(is)", type=["csv", "zip"], accept_multiple_files=True,
        help="E possivel enviar mais de uma parte ou ano.",
    )
    exercicio = st.radio("Exercicio", EXERCICIOS)


# EXERCICIO 1 - RESPOSTA TEXTUAL: IMPORTANCIA DA VISUALIZACAO DE DADOS
if exercicio.startswith("1."):
    st.header("1. Importancia da visualizacao de dados em uma pandemia")
    st.markdown(
        """
        A visualizacao transforma grandes volumes de registros epidemiologicos em
        informacoes compreensiveis. Graficos e mapas ajudam a identificar o crescimento
        ou a queda de casos, comparar territorios, reconhecer grupos mais afetados e
        acompanhar a pressao sobre os servicos de saude.

        Para gestores, essas visualizacoes apoiam a distribuicao de equipes, leitos,
        testes, medicamentos e campanhas de prevencao, alem da avaliacao de medidas ja
        adotadas. Para a populacao, comunicam o nivel de risco, ajudam a combater a
        desinformacao e orientam escolhas de protecao. A leitura deve considerar a
        qualidade dos registros, atrasos de notificacao e o fato de que correlacao nao
        demonstra, sozinha, uma relacao de causa e efeito.
        """
    )
    st.info("Escolha outro exercicio na barra lateral para abrir uma visualizacao.")
    st.stop()

if not arquivos:
    st.info("Envie na barra lateral o ZIP ou os CSVs baixados do painel oficial.")
    st.stop()

try:
    with st.spinner("Preparando os dados oficiais..."):
        dados = carregar_dados(tuple((a.name, a.getvalue()) for a in arquivos))
except Exception as erro:
    st.error(str(erro))
    st.stop()

estaduais, municipais = linhas_estaduais(dados), linhas_municipais(dados)
ufs = sorted(estaduais["estado"].dropna().unique().tolist())
if estaduais.empty:
    st.error("Nao foram encontradas linhas estaduais no arquivo enviado.")
    st.stop()
st.caption(
    f"Base preparada: {len(dados):,} linhas relevantes, de "
    f"{dados['data'].min().date()} a {dados['data'].max().date()}."
)


# EXERCICIO 2 - GRAFICO DE BARRAS COM STREAMLIT
if exercicio.startswith("2."):
    st.header("2. Casos novos por semana epidemiologica - Streamlit")
    c1, c2 = st.columns(2)
    with c1:
        uf = selecionar("Estado", ufs, "RJ", "uf2")
    anos = sorted(estaduais.loc[estaduais["estado"] == uf, "ano"].dropna().astype(int).unique())
    with c2:
        ano = st.selectbox("Ano", anos, index=len(anos) - 1, key="ano2")
    serie = semanal_novos(estaduais.loc[(estaduais["estado"] == uf) & (estaduais["ano"] == ano)], [])
    st.bar_chart(serie, x="periodo_semana", y="casos_novos", color="#1f77b4")
    st.markdown(
        f"**Estado escolhido: {uf}.** O RJ e o padrao por reunir uma grande regiao "
        "metropolitana e municipios do interior. Barras altas indicam semanas com mais notificacoes."
    )


# EXERCICIO 3 - GRAFICO DE LINHA COM STREAMLIT
elif exercicio.startswith("3."):
    st.header("3. Obitos acumulados no Brasil - Streamlit")
    brasil = dados.loc[
        dados["estado"].isna() & dados["municipio"].isna()
        & (dados["regiao"].astype("string").str.lower() == "brasil")
    ]
    if not brasil.empty:
        serie = brasil.groupby(["ordem_semana", "periodo_semana"], as_index=False).agg(
            data_inicio=("data", "min"), obitos_acumulados=("obitos_acumulados", "max")
        ).sort_values("ordem_semana")
    else:
        por_uf = estaduais.groupby(
            ["estado", "ordem_semana", "periodo_semana"], as_index=False
        ).agg(data_inicio=("data", "min"), obitos_acumulados=("obitos_acumulados", "max"))
        serie = por_uf.groupby(["ordem_semana", "periodo_semana"], as_index=False).agg(
            data_inicio=("data_inicio", "min"), obitos_acumulados=("obitos_acumulados", "sum")
        ).sort_values("ordem_semana")
    st.line_chart(serie, x="data_inicio", y="obitos_acumulados", color="#b22222")
    st.markdown(
        "A curva acumulada nao diminui. Uma inclinacao maior indica crescimento mais rapido; "
        "quando a linha se achata, o ritmo de novos obitos diminuiu. Os pontos finais podem "
        "mudar devido a atrasos de notificacao."
    )


# EXERCICIO 4 - GRAFICO DE AREA COM STREAMLIT
elif exercicio.startswith("4."):
    st.header("4. Casos acumulados em tres estados - Streamlit")
    padrao = [uf for uf in ["SP", "RJ", "MG"] if uf in ufs]
    escolhidos = st.multiselect(
        "Escolha exatamente tres estados", ufs, default=padrao, max_selections=3, key="ufs4"
    )
    if len(escolhidos) != 3:
        st.warning("Selecione tres estados.")
        st.stop()
    area = (
        estaduais.loc[estaduais["estado"].isin(escolhidos)]
        .groupby(["estado", "ordem_semana"], as_index=False)
        .agg(data_inicio=("data", "min"), casos_acumulados=("casos_acumulados", "max"))
        .pivot(index="data_inicio", columns="estado", values="casos_acumulados").sort_index()
    )
    st.area_chart(area)
    finais = area.ffill().iloc[-1].sort_values(ascending=False)
    st.markdown(
        f"No fim da base, **{finais.index[0]}** tem o maior acumulado entre os tres "
        f"({finais.iloc[0]:,.0f} casos). Populacao, urbanizacao, testagem e notificacao "
        "ajudam a explicar as diferencas; totais absolutos nao medem sozinhos o risco individual."
    )


# EXERCICIO 5 - MAPA COM ST.MAP DO STREAMLIT
elif exercicio.startswith("5."):
    st.header("5. Cinco municipios com mais casos acumulados - Streamlit")
    ufs_municipais = sorted(municipais["estado"].dropna().unique().tolist())
    uf = selecionar("Estado", ufs_municipais, "RJ", "uf5")
    top5 = (
        municipais.loc[municipais["estado"] == uf].sort_values(["data", "casos_acumulados"])
        .groupby("municipio", as_index=False).tail(1).nlargest(5, "casos_acumulados").copy()
    )
    coordenadas, completas = carregar_coordenadas()
    top5["municipio_chave"] = top5["municipio"].map(chave_texto)
    mapa = top5.merge(coordenadas, on=["estado", "municipio_chave"], how="left")
    validos = mapa.dropna(subset=["latitude", "longitude"])
    if not validos.empty:
        st.map(validos, latitude="latitude", longitude="longitude", size="casos_acumulados", zoom=6)
    if len(validos) < len(top5):
        faltam = ", ".join(mapa.loc[mapa["latitude"].isna(), "municipio"].astype(str))
        st.warning(f"Sem coordenadas auxiliares para: {faltam}.")
    st.dataframe(
        top5[["municipio", "casos_acumulados"]].rename(
            columns={"municipio": "Municipio", "casos_acumulados": "Casos acumulados"}
        ), hide_index=True, use_container_width=True,
    )
    st.markdown(
        "O mapa mostra concentracoes espaciais e pode orientar recursos e comparacoes entre "
        "municipios vizinhos. A concentracao deve ser analisada junto com populacao, mobilidade e testagem."
    )
    if not completas:
        st.caption("A fonte on-line de coordenadas falhou; foi usado o apoio local do RJ.")


# EXERCICIO 6 - VISUALIZACAO COM MATPLOTLIB
elif exercicio.startswith("6."):
    st.header("6. Casos novos e obitos novos por estado - Matplotlib")
    ordem = estaduais["ordem_semana"].dropna().max()
    semana = estaduais.loc[estaduais["ordem_semana"] == ordem]
    comparacao = semana.groupby("estado", as_index=False)[["casos_novos", "obitos_novos"]].sum()
    comparacao = comparacao.sort_values("casos_novos", ascending=False)
    periodo = semana["periodo_semana"].dropna().iloc[0]
    fig, eixos = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    eixos[0].bar(comparacao["estado"], comparacao["casos_novos"], color="#3182bd")
    eixos[0].set(title=f"Casos novos por estado - {periodo}", ylabel="Casos")
    eixos[1].bar(comparacao["estado"], comparacao["obitos_novos"], color="#de2d26")
    eixos[1].set(title=f"Obitos novos por estado - {periodo}", ylabel="Obitos")
    for eixo in eixos:
        eixo.grid(axis="y", alpha=.25)
    eixos[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    correlacao = comparacao[["casos_novos", "obitos_novos"]].corr().iloc[0, 1]
    st.markdown(
        f"Em **{periodo}**, a correlacao estadual foi **{correlacao:.2f}**. Uma relacao "
        "positiva e esperada, mas uma unica semana nao considera a defasagem entre diagnostico "
        "e obito e nao demonstra causalidade."
    )


# EXERCICIO 7 - BOXPLOT COM SEABORN
elif exercicio.startswith("7."):
    st.header("7. Distribuicao semanal de casos novos - Seaborn")
    semanal = semanal_regioes(dados)
    distribuicao = semanal.loc[semanal["regiao"].isin(["Norte", "Nordeste", "Sudeste"])]
    fig, eixo = plt.subplots(figsize=(11, 6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=distribuicao, x="regiao", y="casos_novos", hue="regiao", legend=False, ax=eixo)
    eixo.set(xlabel="Regiao", ylabel="Casos novos por semana", title="Norte, Nordeste e Sudeste")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    medianas = distribuicao.groupby("regiao")["casos_novos"].median().sort_values(ascending=False)
    st.markdown(
        f"A maior mediana aparece no **{medianas.index[0]}** ({medianas.iloc[0]:,.0f}). "
        "A linha central e a mediana, a caixa contem metade das semanas e extremos indicam "
        "ondas atipicas. Numeros absolutos tambem refletem o tamanho populacional."
    )


# EXERCICIO 8 - GRAFICO DE AREA COM ALTAIR
elif exercicio.startswith("8."):
    st.header("8. Evolucao dos casos novos em uma regiao - Altair")
    semanal = semanal_regioes(dados)
    regiao = selecionar("Regiao", REGIOES, "Nordeste", "regiao8")
    serie = semanal.loc[semanal["regiao"] == regiao]
    grafico = alt.Chart(serie).mark_area(
        line={"color": "#08519c"}, color="#6baed6", opacity=.65
    ).encode(
        x=alt.X("data_inicio:T", title="Inicio da semana"),
        y=alt.Y("casos_novos:Q", title="Casos novos"),
        tooltip=[alt.Tooltip("periodo_semana:N", title="Semana"),
                 alt.Tooltip("casos_novos:Q", title="Casos", format=",")],
    ).properties(height=430).interactive()
    st.altair_chart(grafico, use_container_width=True)
    pico = serie.loc[serie["casos_novos"].idxmax()]
    st.markdown(
        f"Na regiao **{regiao}**, o maior pico aparece em **{pico['periodo_semana']}**, "
        f"com {pico['casos_novos']:,.0f} casos. Picos representam ondas, mas semanas "
        "recentes podem estar incompletas por atraso de notificacao."
    )


# EXERCICIO 9 - HEATMAP DE CORRELACAO COM ALTAIR
elif exercicio.startswith("9."):
    st.header("9. Correlacao entre indicadores - Altair")
    uf = selecionar("Estado", ufs, "RJ", "uf9")
    base = estaduais.loc[estaduais["estado"] == uf]
    indicadores = ["casos_novos", "obitos_novos"]
    if "leitos_ocupados" in base and base["leitos_ocupados"].notna().any():
        indicadores.append("leitos_ocupados")
    matriz = base[indicadores].corr()
    longa = matriz.rename_axis("x").reset_index().melt(id_vars="x", var_name="y", value_name="r")
    rotulos = {"casos_novos": "Casos novos", "obitos_novos": "Obitos novos",
               "leitos_ocupados": "Leitos ocupados"}
    longa["x"], longa["y"] = longa["x"].map(rotulos), longa["y"].map(rotulos)
    base_alt = alt.Chart(longa).encode(x=alt.X("x:N", title=None), y=alt.Y("y:N", title=None))
    cores = base_alt.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue", reverse=True))
    )
    textos = base_alt.mark_text(fontSize=16).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.55", alt.value("white"), alt.value("black")),
    )
    st.altair_chart((cores + textos).properties(height=430), use_container_width=True)
    r = matriz.loc["casos_novos", "obitos_novos"]
    nota = (
        "Os leitos foram incluidos porque existem na base."
        if "leitos_ocupados" in indicadores else
        "O arquivo nao possui leitos ocupados; conforme o enunciado, foram usados os indicadores disponiveis."
    )
    st.markdown(
        f"Em **{uf}**, a correlacao no mesmo dia foi **{r:.2f}**. A defasagem entre "
        f"diagnostico e obito pode enfraquecer essa relacao. {nota}"
    )


# EXERCICIO 10 - GRAFICO DE PIZZA COM PLOTLY
elif exercicio.startswith("10."):
    st.header("10. Distribuicao dos casos acumulados por regiao - Plotly")
    regionais = linhas_regionais(dados)
    if not regionais.empty:
        totais = regionais.sort_values("data").groupby("regiao", as_index=False).tail(1)
        totais = totais[["regiao", "casos_acumulados"]]
    else:
        ultimos = estaduais.sort_values("data").groupby("estado", as_index=False).tail(1)
        totais = ultimos.groupby("regiao", as_index=False)["casos_acumulados"].sum()
    totais = totais.loc[totais["regiao"].isin(REGIOES)]
    figura = go.Figure(go.Pie(
        labels=totais["regiao"], values=totais["casos_acumulados"], hole=.25,
        textinfo="label+percent", hovertemplate="%{label}<br>%{value:,.0f} casos<extra></extra>",
    ))
    figura.update_layout(title="Participacao das cinco regioes")
    st.plotly_chart(figura, use_container_width=True)
    lider = totais.loc[totais["casos_acumulados"].idxmax()]
    percentual = lider["casos_acumulados"] / totais["casos_acumulados"].sum() * 100
    st.markdown(
        f"O **{lider['regiao']}** concentra a maior parcela: **{percentual:.1f}%**. "
        "O total absoluto acompanha em parte a populacao; taxas por habitante comparam melhor o risco."
    )


# EXERCICIO 11 - SUBPLOTS COM PLOTLY
elif exercicio.startswith("11."):
    st.header("11. Casos e obitos novos em duas regioes - Plotly")
    semanal = semanal_regioes(dados)
    escolhidas = st.multiselect(
        "Escolha duas regioes", REGIOES, default=["Sudeste", "Nordeste"],
        max_selections=2, key="reg11",
    )
    if len(escolhidas) != 2:
        st.warning("Selecione duas regioes.")
        st.stop()
    figura = make_subplots(rows=1, cols=2, subplot_titles=escolhidas)
    for coluna, regiao in enumerate(escolhidas, 1):
        serie = semanal.loc[semanal["regiao"] == regiao]
        figura.add_trace(go.Bar(
            x=serie["periodo_semana"], y=serie["casos_novos"], name="Casos novos",
            marker_color="#3182bd", showlegend=coluna == 1), row=1, col=coluna)
        figura.add_trace(go.Bar(
            x=serie["periodo_semana"], y=serie["obitos_novos"], name="Obitos novos",
            marker_color="#de2d26", showlegend=coluna == 1), row=1, col=coluna)
    figura.update_layout(height=560, barmode="group", title="Comparacao semanal")
    figura.update_xaxes(title_text="Semana", tickangle=-45)
    figura.update_yaxes(title_text="Quantidade")
    st.plotly_chart(figura, use_container_width=True)
    totais = semanal.loc[semanal["regiao"].isin(escolhidas)].groupby("regiao")[["casos_novos"]].sum()
    st.markdown(
        f"**{totais['casos_novos'].idxmax()}** registra mais casos no periodo. Os subplots "
        "comparam o momento das ondas; obitos tendem a responder com atraso e em escala menor."
    )


# EXERCICIO 12 - MAPA INTERATIVO COM PYDECK
elif exercicio.startswith("12."):
    st.header("12. Casos ajustados pela populacao municipal - PyDeck")
    regiao = selecionar("Regiao", REGIOES, "Sudeste", "regiao12")
    base = municipais.loc[municipais["regiao"] == regiao].copy()
    base["municipio_chave"] = base["municipio"].map(chave_texto)
    coordenadas, completas = carregar_coordenadas()
    mapa = base.merge(coordenadas, on=["estado", "municipio_chave"], how="left")
    mapa = mapa.dropna(subset=["latitude", "longitude", "populacao", "casos_acumulados"])
    mapa = mapa.loc[mapa["populacao"] > 0].copy()
    mapa["incidencia_100k"] = mapa["casos_acumulados"] / mapa["populacao"] * 100_000
    mapa["elevacao"] = mapa["incidencia_100k"].clip(0, 150_000)
    if mapa.empty:
        st.warning("Nao ha coordenadas suficientes para a regiao escolhida.")
        st.stop()
    centros = {
        "Norte": (-4.5, -60., 3.2), "Nordeste": (-9., -39., 4.),
        "Sudeste": (-21., -44., 4.2), "Sul": (-27., -51., 4.4),
        "Centro-Oeste": (-15., -55., 3.8),
    }
    latitude, longitude, zoom = centros[regiao]
    camada = pdk.Layer(
        "ColumnLayer", data=mapa, get_position="[longitude, latitude]",
        get_elevation="elevacao", elevation_scale=1, radius=6000,
        get_fill_color=[230, 85, 45, 180], pickable=True, auto_highlight=True,
    )
    deck = pdk.Deck(
        layers=[camada],
        initial_view_state=pdk.ViewState(
            latitude=latitude, longitude=longitude, zoom=zoom, pitch=45, bearing=0
        ),
        tooltip={"html": "<b>{municipio} - {estado}</b><br/>Casos: {casos_acumulados}"
                           "<br/>Incidencia/100 mil: {incidencia_100k}"},
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.markdown(
        "A altura representa a **incidencia acumulada por 100 mil habitantes**. O ajuste "
        "torna municipios de tamanhos diferentes comparaveis. Densidade, mobilidade, moradia, "
        "prevencao e testagem influenciam a disseminacao."
    )
    st.caption(
        "Coordenadas auxiliares: Municipios-Brasileiros (GitHub). "
        "Populacao e casos: Ministerio da Saude."
    )
    if not completas:
        st.warning("A fonte de coordenadas falhou; o mapa ficou limitado ao apoio local do RJ.")
