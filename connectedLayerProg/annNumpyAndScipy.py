# Ce fichier contient les etapes d'un reseau neuronal avec 3 couches cachees

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

N = 100 # nombre de donnees pour chaque classe
d0 = 2 # nombre de dimension
C = 3 # nombre de classes
X = np.zeros((d0, N*C)) # initialiser la matrice de donnees
y = np.zeros(N*C, dtype='uint8') # vecteur de classe

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # rayon des donnees
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # l'angle pour les donnees
  # Stocker les donnees dans une matrice et un vecteur
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j
  
  
# Visualiser les donnees generees :

plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7);
plt.axis('off') # Enlever les axes des donnees
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

# Visualizer et sauvegarder la figure contruite
plt.savefig('EX.png', bbox_inches='tight', dpi = 600)
plt.show()

# Definition la fonction de softmax pour calculer le vecteur output
def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-hot coding methode pour definir que une classe soit un vecteur
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# Definition de la fonction de perte de Cross Entropy (La moyenne des pertes)
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

d0 = 2
d1 = h = 100 # la taille de la chouche cachee 1
d2 = 50 # la taille de la chouche cachee 1
d3 = C = 3 # la taille de la chouche sortie 1
# intitialisation des parametres du probleme
W1 = 0.01*np.random.randn(d0, d1) # la matrice de poids de la premiere couche
b1 = np.zeros((d1, 1)) # le vecteur de biais de la premiere couche
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))
W3 = 0.01*np.random.randn(d2, d3)
b3 = np.zeros((d3, 1))
Y = convert_labels(y, C) # One-hot coding conversion
N = X.shape[1]
eta = 0.01 # pas d'iteration
lossVal = []
for i in range(100000):
    ## l'etape de feedforward
    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.maximum(Z2, 0)
    Z3 = np.dot(W3.T, A2) + b3
    Yhat = softmax(Z3)

    # affichage de perte apres chaque 100 iteration
    if i %500 == 0:
        # calcul de perte: perte de cross-entropy moyenne
        loss = cost(Y, Yhat)
        lossVal.append(loss)
        print("iteration %d, perte: %f" %(i, loss))

    # l'etape de backpropagation
    E3 = (Yhat - Y )/N
    dW3 = np.dot(A2, E3.T)
    db3 = np.sum(E3, axis = 1, keepdims = True)
    E2 = np.dot(W3, E3)
    E2[Z2 <= 0] = 0 # ReLU
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis = 1, keepdims = True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0 # ReLU
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis = 1, keepdims = True)

    # La mise a jour de
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2
    W3 += -eta*dW3
    b3 += -eta*db3

# La trace de perte apres chaque 200 iterations
plt.figure()
plt.plot(lossVal)
plt.xlabel("Nombre d'iterations / 500")
plt.ylabel("Valeur de perte (%)")
plt.title("La trace de perte pour chaque 500 iterations")

# Test le modele
Z1 = np.dot(W1.T, X) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
A2 = np.maximum(Z2, 0)
Z3 = np.dot(W3.T, A2) + b3
predicted_class = np.argmax(Z3, axis=0)
print('Precision du training: %.2f %%' % (100*np.mean(predicted_class == y)))

# Apres avoir tester avec 2 couches cachees et 3 couches cachees, les resultats
# obtenus sont les memes a 99.33%.

# Tester sur un autre ensemble 
N = 100 # nombre de donnees pour chaque classe
d0 = 2 # nombre de dimension
C = 3 # nombre de classes
X1 = np.zeros((d0, N*C)) # initialiser la matrice de donnees
y1 = np.zeros(N*C, dtype='uint8') # vecteur de classe

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # rayon des donnees
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # l'angle pour les donnees
  # Stocker les donnees dans une matrice et un vecteur
  X1[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y1[ix] = j
  
Z1 = np.dot(W1.T, X1) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
A2 = np.maximum(Z2, 0)
Z3 = np.dot(W3.T, A2) + b3
predicted_class = np.argmax(Z3, axis=0)
print('Precision du test: %.2f %%' % (100*np.mean(predicted_class == y1)))