import numpy as np 

class function_template:
    def __init__(self, start_point, nazwa_funkcji="Testowa"):
        self.start_point=np.array(start_point, dtype=float)
        self. nazwa_funkcji=nazwa_funkcji
    def fval (self, x):
        raise NotImplementedError("Zaimplementuj metodę w klasie dziedziczącej")
    
    def grad_val (self, x):
        raise NotImplementedError("Zaimplementuj metodę w klasie dziedziczącej")
    

#najprostsza funkcja do sprawdzenia czy w ogóle działa
class Sphere(function_template):
    def fval(self, x):
        return np.sum(x**2)

    def grad_val (self, x):
        return 2*x

class Rosenbrock(function_template):
    def __init__(self, start_point=np.array([-5,10]), a=1.0, b=100.0):
        super().__init__(start_point,nazwa_funkcji="Rosenbrocka")
        self.a=a
        self.b=b 
    
    def fval(self, x):

        x1, x2=x[0],x[1]

        return (self.a -x1)**2+self.b*(x2-x1**2)**2

    def grad_val(self, x):

        x1, x2=x[0],x[1]

        dx1 = -2 * (self.a-x1) - 4*self.b*x1*(x2-x1**2)
        dx2 = 2*self.b * (x2-x1**2)

        return np.array([dx1,dx2])

class Three_Hump(function_template):
    def __init__(self, start_point=np.array([-5,5])):
        super().__init__(start_point,nazwa_funkcji="Three_Hump")
        
    
    def fval(self, x):

        x1, x2=x[0],x[1]

        return 2*x1**2-1.05*x1**4+(x1**6)/6+x1*x2+x2**2

    def grad_val(self, x):

        x1, x2=x[0],x[1]

        dx1 = 4*x1-4.2*x1**3+x1**5+x2
        dx2 = x1+2*x2

        return np.array([dx1,dx2])

class Himmelblau(function_template):
    def __init__(self, start_point=np.array([-5,5])):
        super().__init__(start_point,nazwa_funkcji="Himmelblau")
        
    
    def fval(self, x):

        x1, x2=x[0],x[1]

        return (x1**2+x2-11)**2+(x1+x2**2-7)**2

    def grad_val(self, x):

        x1, x2=x[0],x[1]

        dx1 = 4*x1*(x1**2+x2-11)+2*(x1+x2**2-7)
        dx2 = 2*(x1**2+x2-11)+4*x2*(x1+x2**2-7)

        return np.array([dx1,dx2])

class Wartosc_Kary(function_template):
    def __init__(self, a, b, c, nazwa="Ograniczenie"):
        super().__init__(start_point=np.array([0, 0]),nazwa_funkcji=nazwa)
        self.a=float(a)
        self.b=float(b)
        self.c=float(c)

    def fval(self,x):

        x1, x2= x[0],x[1]

        return self.a*x1 + self.b*x2 + self.c

    def grad_val(self,x):
        
        return np.array([self.a,self.b])

class Funkcja_Z_Kara(function_template):
    def __init__(self, target_function, g_constrains=None, h_constrains=None, r=1):
        super().__init__(target_function.start_point, nazwa_funkcji=f"Kara + {target_function.nazwa_funkcji}")
        self.target=target_function
        self.g_constrains = g_constrains if g_constrains is not None else []
        self.h_constrains = h_constrains if h_constrains is not None else []
        self.r = r

    def fval(self,x):

        total_val = self.target.fval(x)
        for g in self.g_constrains:
            val_g = g.fval(x)
            if val_g > 0: 
                total_val += self.r * (val_g**2)
        for h in self.h_constrains:
            val_h = h.fval(x)
            total_val += self.r * (val_h**2) 
        return total_val

    def grad_val(self,x):

        total_grad = self.target.grad_val(x)
        for g in self.g_constrains:
            val_g = g.fval(x) 
            if val_g > 0:
                total_grad += 2 * self.r * val_g * g.grad_val(x)
        for h in self.h_constrains:
            val_h = h.fval(x)
            total_grad += 2 * self.r * val_h * h.grad_val(x)
        return total_grad

class Constraint_Circle(function_template):
    def __init__(self, c=1.0, nazwa="Kolo_Ograniczenie"):
        super().__init__(start_point=np.array([0, 0]), nazwa_funkcji=nazwa)
        self.c = float(c)

    def fval(self, x):
        x1, x2 = x[0], x[1]
        return x1**2 + x2**2 - self.c

    def grad_val(self, x):
        x1, x2 = x[0], x[1]
        return np.array([2*x1, 2*x2])
