#Ejemplo prueba
from dolfin import *
import numpy as np

# Parámetros del problema
a = 3
b = 1.5
c = 0.25
d = 1
ang = 30 * np.pi / 180  # Convertir a radianes

# Definir el dominio usando mshr
from mshr import *

# Cuerpo
x1 = a * np.cos(np.pi - ang)
y1 = b * np.sin(np.pi - ang)
ellipse = Ellipse(Point(0, 0), a, b, np.pi - ang, np.pi + ang)

# Cola
segment1 = Polygon([Point(x1, y1), Point(x1 - d, -x1 + y1 + x1)])
segment2 = Polygon([Point(x1 - d, -x1 + y1 + x1), Point(x1 - d, y1 + d), Point(x1 - d, -x1 + y1 + x1)])
segment3 = Polygon([Point(x1 - d, y1 + d), Point(x1, y1)])

# Ojo
eye = Circle(Point(0.6 * a, 0.4 * b), c)

# Dominio total
domain = ellipse + segment1 + segment2 + segment3 - eye

# Generar malla
mesh = generate_mesh(domain, 80)

# Espacio funcional
V = FunctionSpace(mesh, 'P', 1)

# Condiciones de frontera
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Definir el problema variacional
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=2)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Resolver el problema
u = Function(V)
solve(a == L, u, bc)

# Graficar solución
import matplotlib.pyplot as plt
c = plot(u)
plt.colorbar(c)
plt.show()

# Verificar valores específicos
print("Verificando u(0.5,0.5) =", u(Point(0.5, 0.5)))
print("Verificando u(0.23,0.15) =", u(Point(0.23, 0.15)))
