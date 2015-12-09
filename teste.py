from theano import tensor as T, function, printing

"""x = T.dvector()
hello_world_op = printing.Print('hello world')
printed_x = hello_world_op(x)
f = function([x], printed_x)
r = f([1, 2, 3])"""

x = T.scalar()
y = T.scalar()

z = x+y

w = z*x

a = T.sqrt(5)
b = T.exp(a)
c = a ** b
d = T.log(c)

num1 = input();
num2 = input();

v = T.dvector()

f2 = function([x, y], x + y)

hello_world_op = printing.Print('aaaa')
printed_vetor = hello_world_op(v)
f = function([v], printed_vetor)
r = f([num1, num2])
m = f2(num1, num2)

import mlp
print ('testendo mlp')
mlp.test_mlp()
# treino 80 teste 20

print m