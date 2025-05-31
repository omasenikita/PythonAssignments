class Arithmetic:
    def __init__(self):
        
        self.Ans = 0.0
        self.Value1 = 0
        self.Value2 = 0
        
    def Accept(self,A, B):
        self.Value1 = A
        self.Value2 = B
        
    def Addition(self):
        self.Ans = self.Value1 + self.Value2
        return self.Ans
    
    def Subtraction(self):
        self.Ans = self.Value1 - self.Value2
        return self.Ans
    
    def Multiplication(self):
        self.Ans = self.Value1 * self.Value2
        return self.Ans
    
    def Division(self):
        if(self.Value2 == 0):
            return -1
        
        self.Ans = self.Value1 / self.Value2
        return self.Ans
    
def main():
    val1 = val2 = 0
    
    print("Enter a first value: ")
    val1 = int(input())
    
    print("Enter a second value: ")
    val2 = int(input())
    
    obj1 = Arithmetic()
    
    obj1.Accept(val1,val2)
    
    iret = obj1.Addition()
    print("Addition is",iret)
    
    iret2 = obj1.Subtraction()
    print("Subtraction is ",iret2)
    
    iret3 = obj1.Multiplication()
    print("Multiplication is ",iret3)
    
    iret4 = obj1.Division()
    print("Divistion is",iret4)    

if __name__ =="__main__":
    main()
        