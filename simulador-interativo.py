import pickle

modelo = open('arquivos/modelo_consumo_cerveja', 'rb')
lm_new = pickle.load(modelo)

temp_max = float(input('Qual a temp Máx?'))
chuva = float(input('Qual o valor de chuva?'))
fds = float(input('0 para dias da semana 1 para fins de semana'))

entrada = [[temp_max, chuva, fds]]
print(f'{lm_new.predict(entrada)[0]} litros')