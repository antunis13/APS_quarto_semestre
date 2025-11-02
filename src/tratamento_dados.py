import os
from time import sleep

import pandas as pd
from pandas.core.frame import DataFrame

def tratamento_dados(df: DataFrame):

    pass


def ordenar_lista(lista: list) -> list:

    lista_ordenada = list()

    for valor in (lista):

        valor =  int(valor[6:10])

        if not lista_ordenada:
            lista_ordenada.append(valor)
            continue

        if valor < 0:
            lista_ordenada.insert(0, valor)
        else:
            for index, valor_lista_ordenada in enumerate(lista_ordenada):
                if (index+1) < len(lista_ordenada):
                    if index == 0 and valor < valor_lista_ordenada:
                        lista_ordenada.insert(0, valor)
                        break
                    elif valor > valor_lista_ordenada and valor < lista_ordenada[index+1]:
                        lista_ordenada.insert(index+1, valor)
                        break
                    
                else:
                    lista_ordenada.append(valor)
                    break
    
    lista.clear()

    for index, valor in enumerate(lista_ordenada):
        del lista_ordenada[index]
        lista_ordenada.insert(index, f'dados_{valor}.csv')

    return lista_ordenada


def abrir_arquivos_csv():
    path = os.getcwd() + '/dbqueimadas_CSV'

    arquivos_csv = os.listdir(path=path)

    arquivos_csv = ordenar_lista(lista=arquivos_csv)

    return arquivos_csv


if __name__ == '__main__':
    
    abrir_arquivos_csv()
