import streamlit as st
import pandas as pd

data = {
    "Ano": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Focos_de_Queimada": [21500, 18200, 19800, 23400, 29700, 42300, 38200, 35500, 40100, 37600, 34800]
}

# Dados fictícios de focos de queimadas na Amazônia (2014–2024)
dataMap = {
    "Ano": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Focos_de_Queimada": [21500, 18200, 19800, 23400, 29700, 42300, 38200, 35500, 40100, 37600, 34800],
    "Latitude": [
        -3.4653,  # 2014
        -3.8901,  # 2015
        -4.1023,  # 2016
        -4.5312,  # 2017
        -5.0024,  # 2018
        -5.1243,  # 2019
        -4.7805,  # 2020
        -3.9520,  # 2021
        -3.6725,  # 2022
        -4.2201,  # 2023
        -4.0109,  # 2024
    ],
    "Longitude": [
        -60.0231,  # 2014
        -59.8402,  # 2015
        -60.4523,  # 2016
        -61.0025,  # 2017
        -61.3210,  # 2018
        -62.0341,  # 2019
        -61.6782,  # 2020
        -60.8901,  # 2021
        -60.2312,  # 2022
        -61.0023,  # 2023
        -60.5029,  # 2024
    ]
}

df = pd.DataFrame(data)

dfm = pd.DataFrame(dataMap)


st.title("Queimadas na Amazônia")

st.markdown("""
    Olhar para o tópico de queimadas e focos de incêndio pelo nosso território é de suma importância para a preservação do meio ambiente e da saúde da sociedade. 
    Abaixo analisaremos dados reunidos por mais de uma década, afim de ilustrar a situação em que se encontra o bioma amazônico.
""")

c1 = st.container(border=True, gap="medium")

c1.subheader("Gráfico da quantidade de focos de fogo no período entre 2014 e 2024")

c1.line_chart(df)

c2 = st.container()

c2.subheader("Mapa")

c2.text("Distribuição geográfica")

c2.map(dfm, latitude="Latitude", longitude="Longitude")

c3 = st.container(gap="medium")

c3.subheader("Relatórios")

message = c3.chat_message("Relatórios")

message.write("""
    What is Lorem Ipsum?
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

    Why do we use it?
        It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


    Where does it come from?
        Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.
""")


