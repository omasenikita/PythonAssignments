class Circle:
    PI = 3.14
    def __init__(self):
        self.Area = 0.0
        self.Circumference = 0.0
        self.Radius = 0.0
        
    def Accept(self,A):
        self.Radius = A 
    def CalculateArea(self):
        self.Area = Circle.PI * self.Radius *self.Radius
        return self.Area
        
    def CalculateCircumference(self):
        self.Circumference = 2 * Circle.PI * self.Radius 
        return self.Circumference
        
    def Display(self):
        print("Area is ",self.Area)
        print("Circumference is : ",self.Circumference)
        
        
def main():
    obj1 = Circle()
    obj1.Accept(4)
    obj1.CalculateArea()
    obj1.CalculateCircumference()
    obj1.Display()
    
    obj2 = Circle()
    obj2.Accept(6)
    obj2.CalculateArea()
    obj2.CalculateCircumference()
    obj2.Display()
    
if __name__ =="__main__":
    main()
        
        
    