import random
import matplotlib.pyplot as plt

# Definimos las 3 transformaciones afines para Sierpinski:
funcs = [
    lambda x, y: (0.5*x      , 0.5*y     ),  # escala 1/2 en origen
    lambda x, y: (0.5*x + 0.5, 0.5*y     ),  # escala + traslación a la derecha
    lambda x, y: (0.5*x + 0.25, 0.5*y + 0.5)  # escala + traslación “hacia arriba”
]

# Número de iteraciones y “quemar” los primeros 10 iter
N = 50000
x, y = 0.0, 0.0
xs, ys = [], []

for i in range(N + 10):
    f = random.choice(funcs)
    x, y = f(x, y)
    if i >= 10:
        xs.append(x)
        ys.append(y)

plt.scatter(xs, ys, s=0.1, color="black")
plt.axis("equal")
plt.axis("off")
plt.show()
