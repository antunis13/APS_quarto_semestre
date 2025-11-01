# Projeto de Classificação de Dados do INPE com Rotulagem Manual e Random Forest

## 🎯 Objetivo do Projeto

Este projeto tem como objetivo principal demonstrar um pipeline de Machine Learning que combina a flexibilidade da rotulagem manual de dados com a robustez de um modelo de classificação avançado. Especificamente, ele aborda:

-   **Leitura de Dados**: Utiliza a biblioteca padrão `csv` do Python para carregar dados brutos do INPE.
-   **Pré-processamento e Limpeza**:
    -   Remove registros que contêm valores inválidos (representados por `999`) em colunas de features numéricas.
    -   Filtra o dataset para incluir apenas amostras pertencentes ao bioma **Amazônia**.
    -   Converte as features numéricas para o tipo `float`.
-   **Rotulagem Manual com TADs**:
    -   Emprega Tipos Abstratos de Dados (TADs) — **Fila (Queue)** — para gerenciar o processo de rotulagem manual.
    -   A Fila é usada para enfileirar exemplos que precisam de um rótulo.
-   **Divisão de Dados**: Após a rotulagem, o conjunto de dados é dividido em 80% para treinamento do modelo e 20% para teste, garantindo uma avaliação imparcial.
-   **Treinamento e Avaliação do Modelo**:
    -   Utiliza o algoritmo **Random Forest Progression** da biblioteca `scikit-learn` para treinar um modelo de classificação.
    -   Avalia o desempenho do modelo usando métricas como acurácia, relatório de classificação (precision, recall, F1-score) e matriz de confusão.
    -   Analisa a importância das features para entender quais variáveis mais contribuem para as previsões do modelo.

Este projeto é ideal para cenários onde a qualidade dos rótulos é crítica e exige intervenção humana, ao mesmo tempo em que se beneficia de um classificador poderoso.

---

## 🚀 Como Ativar o Ambiente Virtual (venv)

É altamente recomendável usar um ambiente virtual para isolar as dependências do projeto.

### Pré-requisitos

-   **Python 3.9+** (versão recomendada)
-   **pip** (gerenciador de pacotes do Python) atualizado:
    ```bash
    python -m pip install --upgrade pip
    ```
-   **/dbqueimadas_CSV** (pasta de CSVs com os dados de treinamento)
    Para que seja possível você criar o seu modelo de ML usando **Random Forest Progression** você tem que criar uma pasta na raiz do projeto chamada **dbqueimadas_CSV** e colocar os seus arquivos CSV dentro dela.

### 💻 macOS e Linux

1.  **Criar o ambiente virtual**:
    ```bash
    python3 -m venv .venv
    ```
2.  **Ativar o ambiente virtual**:
    ```bash
    source .venv/bin/activate
    ```
3.  **Desativar o ambiente virtual** (quando terminar de trabalhar no projeto):
    ```bash
    deactivate
    ```

### 🖥️ Windows (PowerShell)

1.  **Criar o ambiente virtual**:
    ```powershell
    python -m venv .venv
    ```
2.  **Ativar o ambiente virtual**:
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
    *Se você encontrar um erro de execução de script, pode ser necessário ajustar a política de execução do PowerShell. Abra o PowerShell como **Administrador** e execute:*
    ```powershell
    Set-ExecutionPolicy RemoteSigned
    ```
    *Após isso, tente ativar o ambiente virtual novamente.*

3.  **Desativar o ambiente virtual**:
    ```powershell
    deactivate
    ```

---

## 📦 Bibliotecas Utilizadas

As seguintes bibliotecas Python são necessárias para executar este projeto:

-   **`scikit-learn`**: Para a implementação do modelo Random Forest e métricas de avaliação.
-   **`pandas`**: Embora a leitura inicial use `csv` padrão, `pandas` é uma ferramenta poderosa para manipulação e análise de dados, sendo uma dependência comum em projetos de ML.
-   **`numpy`**: Biblioteca fundamental para computação numérica em Python, base para `pandas` e `scikit-learn`.
-   **`matplotlib`**: Para a criação de gráficos e visualizações (ex: matriz de confusão, importância de features).
-   **`seaborn`**: Baseado em `matplotlib`, oferece uma interface de alto nível para criar gráficos estatísticos atraentes e informativos.

### Instalação das Bibliotecas

Com o ambiente virtual ativado, instale todas as dependências de uma vez:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn