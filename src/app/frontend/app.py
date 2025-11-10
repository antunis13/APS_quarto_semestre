import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
from preditor import PredictorQueimadas


def criar_mapa_sensorial(resultados):
    """
    Cria um mapa sensorial com cores baseadas no risco.
    
    Args:
        resultados: DataFrame com predições e coordenadas
    """
    
    # Criar mapa centrado na Amazônia
    mapa = folium.Map(
        location=[-4.0, -60.0],
        zoom_start=5,
        tiles="OpenStreetMap"
    )
    
    # Mapear cores por categoria
    cores_mapa = {
        'Baixo': 'green',
        'Médio': 'orange',
        'Alto': 'red'
    }
    
    # Adicionar marker para cada município
    for idx, row in resultados.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        categoria = row['categoria']
        cor = cores_mapa.get(categoria, 'gray')
        
        # Criar popup com informações
        popup_text = f"""
        <b>{row['municipio_nome']}</b><br>
        Categoria: <b>{categoria}</b><br>
        Confiança: {row['confianca']:.1%}<br>
        Risco Fogo: {row['risco_fogo']:.2f}<br>
        Dias sem Chuva: {row['dias_sem_chuva']:.1f}<br>
        Precipitação: {row['precipitacao']:.2f}
        """
        
        # Adicionar círculo colorido (maior que marker padrão)
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color=cor,
            fill=True,
            fillColor=cor,
            fillOpacity=0.7,
            weight=2,
            opacity=0.8
        ).add_to(mapa)
    
    return mapa


# Inicializar preditor (cache para não recarregar toda vez)
@st.cache_resource
def load_predictor():
    return PredictorQueimadas(
        caminho_modelo='modelo_RF.jkl',
        caminho_dados='dbqueimadas_CSV/df_final.csv'
    )


# Configuração da página
st.set_page_config(
    page_title="Predição de Queimadas",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Predição de Risco de Queimadas na Amazônia")

predictor = load_predictor()

# Seleção de data e municípios
st.subheader("Configurações da Predição")

col1, col2 = st.columns(2)

with col1:
    data_pred = st.date_input("📅 Selecione a data para predição")

with col2:
    # Botão para selecionar todos
    if st.button("✅ Selecionar Todos os Municípios", width='stretch'):
        st.session_state.municipios_selecionados = list(predictor.municipios.values())
    
    # Multiselect com state
    municipios_selecionados = st.multiselect(
        "Selecione os municípios",
        list(predictor.municipios.values()),
        default=st.session_state.get('municipios_selecionados', ["Manaus", "Presidente Figueiredo"]),
        key='municipios_selecionados'
    )

# Converter nomes para códigos
municipios_codigos = [cod for cod, nome in predictor.municipios.items() 
                      if nome in municipios_selecionados]

# Botão de predição
if st.button("🚀 Fazer Predição", width='stretch', type="primary"):
    with st.spinner("Fazendo predições... ⏳"):
        data_pred_datetime = datetime(data_pred.year, data_pred.month, data_pred.day)
        st.session_state.resultados = predictor.prever_multiplos_municipios(data_pred_datetime, municipios_codigos)
        st.session_state.data_pred_realizada = data_pred
    st.rerun()

# Mostrar resultados se existirem
if 'resultados' in st.session_state and not st.session_state.resultados.empty:
    st.success("✓ Predições concluídas!")
    resultados = st.session_state.resultados
    
    if not resultados.empty:
        # Criar abas: Mapa, Tabela e Estatísticas
        tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Sensorial", "📊 Tabela de Dados", "📈 Estatísticas"])
        
        with tab1:
            st.subheader("Mapa de Risco de Queimadas")
            mapa = criar_mapa_sensorial(resultados)
            st_folium(mapa, width=1200, height=600)
            
            # Legenda
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **Baixo Risco** - Condições favoráveis")
            with col2:
                st.markdown("🟠 **Médio Risco** - Atenção necessária")
            with col3:
                st.markdown("🔴 **Alto Risco** - Perigo iminente")
        
        with tab2:
            st.subheader("Detalhes das Predições")
            
            # Filtrar colunas importantes
            df_display = resultados[['municipio_nome', 'categoria', 'confianca', 'risco_fogo', 'dias_sem_chuva', 'precipitacao']].copy()
            
            # Formatar confiança como percentual
            df_display['confianca'] = df_display['confianca'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(
                df_display,
                width='stretch',
                hide_index=True
            )
            
            # Opção de download
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Baixar dados como CSV",
                data=csv,
                file_name=f"predicoes_{data_pred}.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("Estatísticas da Predição")
            
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            
            baixos = (resultados['categoria'] == 'Baixo').sum()
            medios = (resultados['categoria'] == 'Médio').sum()
            altos = (resultados['categoria'] == 'Alto').sum()
            
            with col1:
                st.metric(
                    "🟢 Baixo Risco",
                    f"{baixos}",
                    f"{baixos/len(resultados)*100:.1f}%"
                )
            with col2:
                st.metric(
                    "🟠 Médio Risco",
                    f"{medios}",
                    f"{medios/len(resultados)*100:.1f}%"
                )
            with col3:
                st.metric(
                    "🔴 Alto Risco",
                    f"{altos}",
                    f"{altos/len(resultados)*100:.1f}%"
                )
            
            # Gráfico de confiança média por categoria
            st.markdown("---")
            st.markdown("### Confiança Média por Categoria")
            
            confianca_por_categoria = resultados.groupby('categoria')['confianca'].mean().sort_values(ascending=False)
            st.bar_chart(confianca_por_categoria)
            
            # Tabela de estatísticas
            st.markdown("---")
            st.markdown("### Dados Meteorológicos Médios")
            
            stats = pd.DataFrame({
                'Categoria': resultados['categoria'].unique(),
                'Risco Fogo Médio': [resultados[resultados['categoria'] == cat]['risco_fogo'].mean() for cat in resultados['categoria'].unique()],
                'Dias sem Chuva Médio': [resultados[resultados['categoria'] == cat]['dias_sem_chuva'].mean() for cat in resultados['categoria'].unique()],
                'Precipitação Média': [resultados[resultados['categoria'] == cat]['precipitacao'].mean() for cat in resultados['categoria'].unique()],
            })
            
            st.dataframe(stats, width='stretch', hide_index=True)
    
    else:
        st.error("❌ Nenhuma predição foi feita. Tente novamente.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>Sistema de Predição de Queimadas na Amazônia | Modelo: RandomForest | Atualizado: 2025</small>
    </div>
""", unsafe_allow_html=True)