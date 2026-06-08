import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore') # Ignorar avisos chatos do pandas

# ==========================================
# CONFIGURAÇÃO VISUAL DA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard Games", layout="wide")
st.title("🎮 Previsão de Sucesso Comercial de Videogames")
st.markdown("Projeto de Estatística e Probabilidade - Análise de Hits (>1M de cópias)")

# ==========================================
# 1. CARREGAMENTO E LIMPEZA DOS DADOS
# ==========================================
# Usamos o cache do streamlit pra não ter que recarregar o CSV pesado toda vez que apertar um botão
@st.cache_data
def carregar_dados():
    df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
    
    # Criando a nossa variável alvo. Se vendeu mais de 1 milhão, é Hit (1).
    df['Hit_Comercial'] = df['Global_Sales'].apply(lambda x: 1 if x > 1.0 else 0)
    
    # Tinha uns textos 'tbd' misturados na nota dos usuários. Trocamos por nulo e forçamos virar número.
    df['User_Score'] = df['User_Score'].replace('tbd', np.nan).astype(float)
    
    # Jogando fora as linhas que não tem notas. (Se preencher com a média, estraga o Teorema de Bayes)
    df_limpo = df.dropna(subset=['Critic_Score', 'User_Score', 'Rating', 'Genre', 'Platform']).copy()
    
    # Criando faixas de notas pra facilitar o cálculo manual da probabilidade depois
    faixas = [0, 60, 80, 100]
    nomes_faixas = ['Baixa', 'Média', 'Alta']
    df_limpo['Faixa_Critica'] = pd.cut(df_limpo['Critic_Score'], bins=faixas, labels=nomes_faixas)
    
    return df_limpo

dados = carregar_dados()

# ==========================================
# 2. TREINAMENTO DOS MODELOS DE MACHINE LEARNING
# ==========================================
@st.cache_resource
def treinar_modelos(df):
    features = df[['Genre', 'Platform', 'Rating', 'Critic_Score', 'User_Score']]
    # O scikit-learn não entende texto, então isso aqui transforma as categorias em colunas de 0 e 1
    features_numericas = pd.get_dummies(features, drop_first=True)
    alvo = df['Hit_Comercial']
    
    # Separando 30% dos dados para a prova final (teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(features_numericas, alvo, test_size=0.3, random_state=42)
    
    # Treinando a Árvore (limitada a 5 níveis pra não decorar o gabarito)
    arvore = DecisionTreeClassifier(max_depth=5, random_state=42)
    arvore.fit(X_treino, y_treino)
    
    # Treinando a Regressão Logística
    logistica = LogisticRegression(max_iter=1000)
    logistica.fit(X_treino, y_treino)
    
    return arvore, logistica, features_numericas.columns

modelo_arvore, modelo_logistica, colunas_treinadas = treinar_modelos(dados)

# ==========================================
# SEÇÃO 1 - ANÁLISE EXPLORATÓRIA (EDA)
# ==========================================
st.header("📊 Seção 1 - Análise Exploratória de Dados")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Taxa de Hits por Gênero")
    # Calculando a média de sucesso de cada gênero
    agrupado_genero = dados.groupby('Genre')['Hit_Comercial'].mean().reset_index()
    grafico_barras = px.bar(agrupado_genero, x='Genre', y='Hit_Comercial', 
                            title="Qual a chance histórica de um gênero estourar?",
                            labels={'Hit_Comercial': 'Taxa de Hit', 'Genre': 'Gênero'})
    grafico_barras.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(grafico_barras, use_container_width=True)

with col2:
    st.subheader("Nota da Crítica vs Sucesso")
    grafico_caixa = px.box(dados, x='Hit_Comercial', y='Critic_Score', 
                           color='Hit_Comercial', title="Jogos que viram Hits tem notas maiores?",
                           labels={'Hit_Comercial': 'Virou Hit?', 'Critic_Score': 'Nota da Crítica'})
    st.plotly_chart(grafico_caixa, use_container_width=True)

st.divider()

# ==========================================
# SEÇÃO 2 - SIMULADOR PROBABILÍSTICO
# ==========================================
st.header("🔮 Seção 2 - Classificação Probabilística (Simulador)")
st.markdown("Brinque com os dados abaixo para ver o que os nossos modelos acham do seu jogo hipotético.")

colA, colB, colC = st.columns(3)
with colA:
    escolha_genero = st.selectbox("Gênero do Jogo", sorted(dados['Genre'].unique()))
with colB:
    escolha_plataforma = st.selectbox("Plataforma", sorted(dados['Platform'].unique()))
with colC:
    escolha_nota = st.selectbox("Expectativa de Nota da Crítica", ['Baixa', 'Média', 'Alta'])

# ==========================================
# MATEMÁTICA: TEOREMA DE BAYES NA MÃO
# ==========================================
def calcular_bayes_manual(genero, plataforma, nota):
    prob_hit_geral = dados['Hit_Comercial'].mean()
    prob_fracasso_geral = 1 - prob_hit_geral
    
    # Separando quem fez sucesso de quem não fez
    jogos_sucesso = dados[dados['Hit_Comercial'] == 1]
    jogos_fracasso = dados[dados['Hit_Comercial'] == 0]
    
    # Calculando as Verossimilhanças (P(X|C))
    # Colocamos o +1 e o total de opções únicas (Laplace Smoothing) para a conta não dar zero absoluto se o jogo for muito estranho
    total_generos = len(dados['Genre'].unique())
    p_genero_dado_hit = (len(jogos_sucesso[jogos_sucesso['Genre'] == genero]) + 1) / (len(jogos_sucesso) + total_generos)
    p_genero_dado_fracasso = (len(jogos_fracasso[jogos_fracasso['Genre'] == genero]) + 1) / (len(jogos_fracasso) + total_generos)
    
    total_plataformas = len(dados['Platform'].unique())
    p_plat_dado_hit = (len(jogos_sucesso[jogos_sucesso['Platform'] == plataforma]) + 1) / (len(jogos_sucesso) + total_plataformas)
    p_plat_dado_fracasso = (len(jogos_fracasso[jogos_fracasso['Platform'] == plataforma]) + 1) / (len(jogos_fracasso) + total_plataformas)
    
    total_faixas = len(dados['Faixa_Critica'].unique())
    p_nota_dado_hit = (len(jogos_sucesso[jogos_sucesso['Faixa_Critica'] == nota]) + 1) / (len(jogos_sucesso) + total_faixas)
    p_nota_dado_fracasso = (len(jogos_fracasso[jogos_fracasso['Faixa_Critica'] == nota]) + 1) / (len(jogos_fracasso) + total_faixas)
    
    # Multiplicando tudo (Numerador do Teorema)
    parte_cima_hit = p_genero_dado_hit * p_plat_dado_hit * p_nota_dado_hit * prob_hit_geral
    parte_cima_fracasso = p_genero_dado_fracasso * p_plat_dado_fracasso * p_nota_dado_fracasso * prob_fracasso_geral
    
    # Fechando a fórmula (Probabilidade a posteriori final)
    return parte_cima_hit / (parte_cima_hit + parte_cima_fracasso)

# Botão de Ação
if st.button("🚀 Calcular Previsões", type="primary", use_container_width=True):
    
    # 1. Rodando o Bayes manual
    resultado_bayes = calcular_bayes_manual(escolha_genero, escolha_plataforma, escolha_nota)
    
    # 2. Rodando o Machine Learning
    # Como os algoritmos de ML precisam de números e não da palavra "Média", a gente chuta um valor numérico para simular
    chute_nota = 50 if escolha_nota == 'Baixa' else 70 if escolha_nota == 'Média' else 90
    
    # Montando a "tabelinha" do jogo novo exatamente como os modelos estudaram
    jogo_hipotetico = pd.DataFrame({
        'Genre': [escolha_genero], 'Platform': [escolha_plataforma], 
        'Rating': ['E'], # Classificação neutra padrão pra não atrapalhar
        'Critic_Score': [chute_nota], 'User_Score': [chute_nota / 10]
    })
    
    # Aplicando o get_dummies e alinhando com as colunas que o modelo conhece
    jogo_numerico = pd.get_dummies(jogo_hipotetico).reindex(columns=colunas_treinadas, fill_value=0)
    
    # Pegando as previsões
    veredito_arvore = modelo_arvore.predict(jogo_numerico)[0]
    prob_logistica = modelo_logistica.predict_proba(jogo_numerico)[0][1]
    
    # Exibindo tudo na tela
    st.subheader("Resultados Comparativos")
    res1, res2, res3 = st.columns(3)
    
    with res1:
        st.info(f"**Teorema de Bayes**\n\nProbabilidade: **{resultado_bayes:.1%}**")
    with res2:
        texto_final_arvore = "Vai ser um HIT!" if veredito_arvore == 1 else "Vai Fracassar :("
        st.success(f"**Árvore de Decisão**\n\nVeredito: **{texto_final_arvore}**")
    with res3:
        st.warning(f"**Regressão Logística**\n\nProbabilidade: **{prob_logistica:.1%}**")