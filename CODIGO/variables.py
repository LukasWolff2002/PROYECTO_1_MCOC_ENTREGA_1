#Definir variables aqui
num_equipotenciales = 8 #debe ser un numero par
gamma_agua = 9810 #N/m3
gamma_sturada = 21000 #N/m3
altura_rel = 0

#Variables suelo
k = 6.9e-5

#Caso 1
a1 = 0.8
b1 = 3.8
c1 = 18.6
a2 = 10.0
b2 = 3.0
c2 = 10.2
d = 13 - (a2+b2)
caso_1 = {'a1': a1, 'b1': b1, 'c1': c1, 'a2': a2, 'b2': b2, 'c2': c2, 'd': d, 'k': k}

#Caso 2
a1 = 0.8
b1 = 3.8
c1 = 18.6
a2 = 7.6
b2 = 3.0
c2 = 12.6
d = 13 - (a2+b2)
caso_2 = {'a1': a1, 'b1': b1, 'c1': c1, 'a2': a2, 'b2': b2, 'c2': c2, 'd': d, 'k': k}

#Caso 3
a1 = 0.8
b1 = 3.8
c1 = 18.6
a2 = 6.2
b2 = 1.0
c2 = 16.0
d = 13 - (a2+b2)
caso_3 = {'a1': a1, 'b1': b1, 'c1': c1, 'a2': a2, 'b2': b2, 'c2': c2, 'd': d, 'k': k}

#Caso Ejemplo libro
a1 = 0.8
b1 = 4
c1 = 12
a2 = 6.2
b2 = 1.0
c2 = 12
d = 6
caso_ejemplo = {'a1': a1, 'b1': b1, 'c1': c1, 'a2': a2, 'b2': b2, 'c2': c2, 'd': d, 'k': 2e-6}



