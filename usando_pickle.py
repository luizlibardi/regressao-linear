import pickle

modelo = open('arquivos/modelo_consumo_cerveja', 'rb')
lm_new = pickle.load(modelo)

temp_max = 30.5
chuva = 12.2
fds = 0
entrada = [[temp_max, chuva, fds]]
print(f'{lm_new.predict(entrada)[0]} litros')