import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(-np.pi, np.pi, 70) 
y = np.tanh(x) 

plt.plot(x, y) 
plt.xlabel("x") 
plt.ylabel("Tanh(x)") 
plt.savefig("plotare_tanh.png") 
plt.close

