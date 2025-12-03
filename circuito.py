#treinamento rede neural com lei de Ohm e linearização simples com pytorch#

import torch
import torch.nn as nn
import torch.optim as optim

#dados
corrente = [0.01, 0.02, 0.03, 0.04]
tensao = [1.02, 2.05, 3.01, 4.08]

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

print("\n ----- Resultado Final -----")
print(f"Resistência/Coeficiente 'a' : {a.item():.4f} ohms")
print(f"Intercepção 'b':              {b.item():.4f} volts")
print("-----------------------------------------------------")