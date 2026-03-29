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
    def __init__(self, start_point, a=1.0, b=100.0):
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
    def __init__(self, start_point):
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
    def __init__(self, start_point):
        super().__init__(start_point,nazwa_funkcji="Himmelblau")
        
    
    def fval(self, x):

        x1, x2=x[0],x[1]

        return (x1**2+x2-11)**2+(x1+x2**2-7)**2

    def grad_val(self, x):

        x1, x2=x[0],x[1]

        dx1 = 4*x1*(x1**2+x2-11)+2*(x1+x2**2-7)
        dx2 = 2*(x1**2+x2-11)+4*x2*(x1+x2**2-7)

        return np.array([dx1,dx2])