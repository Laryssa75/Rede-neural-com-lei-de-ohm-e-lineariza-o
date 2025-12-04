#treinamento rede neural com lei de Ohm e linearização simples com pytorch#

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#dados
# corrente = [0.01, 0.02, 0.03, 0.04]
# tensao = [1.02, 2.05, 3.01, 4.08]

corrente = [0.00297, 0.00153]
tensao = [1.89, 1.85]

I = torch.tensor(corrente, dtype=torch.float32).unsqueeze(1)
V = torch.tensor(tensao, dtype=torch.float32).unsqueeze(1)

#modelo,  otimizador e perda

#modelo V = a * I + b (1 entrada, 1 saída)
model = nn.Linear(1,1)

#otimizador: Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.1)

#função de perda: Mean Squared Error (MSE)
loss_fn = nn.MSELoss()

#treinamento

#treinamento com 200 épocas (iterações)
print("Inciciando treinamento...")

for epoch in range(200):
    
    #1.Previsão
    pred = model(I)
    
    #2. Cálculo de perda
    loss = loss_fn(pred, V)
    
    #3.Zera os gradientes
    optimizer.zero_grad()
    
    #4.Backward - calcula os gradientes
    loss.backward()
    
    #5.Atualiza os parâmetros (a e b)
    optimizer.step()
    
    #Opicional: imprimir a perda a cada 50 épocas para acompanhar
    if(epoch + 1) % 50 == 0:
        print(f"Época [{epoch + 1 } / 200], Perda: {loss.item():.6f}") 
        
#Resultado final
a, b = model.parameters()
a_val = a.item()
b_val = b.item()

print("\n ----- Resultado Final -----")
print(f"Resistência/Coeficiente 'a' : {a_val:.4f} ohms")
print(f"Intercepção 'b':              {b_val:.4f} volts")
print("-----------------------------------------------------")

#Geração de gráfico

#1.Converte tensores para arrays numpy para plotagem
corrente_np = I.numpy().flatten()
voltagem_np = V.numpy().flatten()

# 2. Calcula a linha de ajuste usando o modelo aprendido
# pred_line = a_val * I_np + b_val
# Também podemos usar o próprio modelo para prever a linha
with torch.no_grad(): # Desativa o cálculo de gradientes para a inferência
    V_pred_np = model(I).numpy().flatten()
    
# 3. Configura a figura e eixos
plt.figure(figsize=(8, 5))
plt.title(f'Linearização Tensão vs. Corrente (V = {a_val:.2f}*I + {b_val:.2f})')
plt.xlabel('Corrente, I (Amperes)')
plt.ylabel('Tensão, V (Volts)')
plt.grid(True)

# 4. Plota os dados originais (pontos dispersos)
plt.scatter(corrente_np, voltagem_np, color='blue', label='Dados Originais (I, V)')

# 5. Plota a linha aprendida pelo modelo
plt.plot(corrente_np, V_pred_np, color='red', linestyle='-', label='Modelo Linear (V = a*I + b)')

# 6. Exibe a legenda e o gráfico
plt.legend()
plt.show()