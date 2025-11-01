import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue, Process
import time

# ============================================
# FUNÇÃO WORKER (PRODUTOR)
# ============================================
def worker_processar_chunk(chunk_info, queue, filtros):
    """
    Processa um chunk do CSV e adiciona resultados filtrados na fila
    
    Args:
        chunk_info: tupla (numero_chunk, chunk_dataframe)
        queue: fila compartilhada para adicionar resultados
        filtros: dicionário com critérios de filtro
    """
    chunk_num, chunk_df = chunk_info
    
    print(f"[Worker {chunk_num}] Processando {len(chunk_df)} linhas...")
    
    # APLICAR FILTROS (exemplo: valor > 500 e status ativo)
    df_filtrado = chunk_df[
        (chunk_df['valor'] > filtros['valor_min']) & 
        (chunk_df['status'] == filtros['status'])
    ].copy()
    
    # TRATAMENTO DE DADOS (exemplo: normalizar score)
    df_filtrado['score_normalizado'] = df_filtrado['score'] / 100
    
    # Converter para lista de dicionários para facilitar serialização
    resultados = df_filtrado.to_dict('records')
    
    # ADICIONAR NA FILA
    queue.put({
        'chunk_num': chunk_num,
        'dados': resultados,
        'total_processado': len(chunk_df),
        'total_filtrado': len(resultados)
    })
    
    print(f"[Worker {chunk_num}] Concluído: {len(resultados)} linhas filtradas")


# ============================================
# FUNÇÃO CONSUMIDOR
# ============================================
def consumidor(queue, num_chunks, resultado_final):
    """
    Consome dados da fila e monta DataFrame final
    
    Args:
        queue: fila para consumir dados
        num_chunks: número total de chunks esperados
        resultado_final: lista compartilhada para armazenar resultados
    """
    chunks_recebidos = 0
    todos_dados = []
    
    print("[Consumidor] Iniciado, aguardando dados...")
    
    while chunks_recebidos < num_chunks:
        # RETIRAR DA FILA (bloqueia até ter dados)
        item = queue.get()
        
        if item is None:  # Poison pill
            break
            
        print(f"[Consumidor] Recebido chunk {item['chunk_num']}: "
              f"{item['total_filtrado']}/{item['total_processado']} linhas")
        
        todos_dados.extend(item['dados'])
        chunks_recebidos += 1
    
    # Montar DataFrame final
    df_final = pd.DataFrame(todos_dados)
    resultado_final.append(df_final)
    
    print(f"[Consumidor] Finalizado! Total de {len(df_final)} linhas no resultado final")


# ============================================
# FUNÇÃO PRINCIPAL
# ============================================
def processar_csv_paralelo(arquivo_csv, filtros, chunk_size=10000, num_workers=4):
    """
    Processa CSV grande usando multiprocessing com Queue
    
    Args:
        arquivo_csv: caminho do arquivo CSV
        filtros: dicionário com critérios de filtro
        chunk_size: tamanho de cada chunk
        num_workers: número de processos workers
    """
    print(f"\n{'='*60}")
    print(f"Iniciando processamento paralelo")
    print(f"Workers: {num_workers} | Chunk size: {chunk_size}")
    print(f"{'='*60}\n")
    
    inicio = time.time()
    
    # CRIAR FILA COMPARTILHADA
    queue = Queue(maxsize=num_workers * 2)  # Limitar tamanho para não sobrecarregar memória
    
    # LER CSV EM CHUNKS
    chunks = []
    for i, chunk in enumerate(pd.read_csv(arquivo_csv, chunksize=chunk_size)):
        chunks.append((i, chunk))
    
    num_chunks = len(chunks)
    print(f"CSV dividido em {num_chunks} chunks\n")
    
    # CRIAR PROCESSO CONSUMIDOR
    manager = mp.Manager()
    resultado_final = manager.list()
    
    processo_consumidor = Process(
        target=consumidor,
        args=(queue, num_chunks, resultado_final)
    )
    processo_consumidor.start()
    
    # CRIAR POOL DE WORKERS PRODUTORES
    processos_workers = []
    
    for chunk_info in chunks:
        # Criar processo para cada chunk
        p = Process(
            target=worker_processar_chunk,
            args=(chunk_info, queue, filtros)
        )
        processos_workers.append(p)
        p.start()
        
        # Limitar número de processos simultâneos
        if len(processos_workers) >= num_workers:
            for proc in processos_workers:
                proc.join()
            processos_workers = []
    
    # Aguardar workers restantes
    for proc in processos_workers:
        proc.join()
    
    # Aguardar consumidor
    processo_consumidor.join()
    
    tempo_total = time.time() - inicio
    
    print(f"\n{'='*60}")
    print(f"Processamento concluído em {tempo_total:.2f} segundos")
    print(f"{'='*60}\n")
    
    # Retornar DataFrame final
    return resultado_final[0] if resultado_final else pd.DataFrame()


# ============================================
# EXEMPLO DE USO
# ============================================
if __name__ == '__main__':
    # Definir filtros
    filtros = {
        'valor_min': 500,
        'status': 'ativo'
    }
    
    # Processar CSV
    df_resultado = processar_csv_paralelo(
        arquivo_csv='dados_exemplo.csv',
        filtros=filtros,
        chunk_size=20000,
        num_workers=4
    )
    
    print("Resultado final:")
    print(df_resultado.head(10))
    print(f"\nShape: {df_resultado.shape}")
    print(f"\nEstatísticas:")
    print(df_resultado.describe())
    
    # Salvar resultado
    df_resultado.to_csv('dados_processados.csv', index=False)
    print("\nArquivo 'dados_processados.csv' criado!")