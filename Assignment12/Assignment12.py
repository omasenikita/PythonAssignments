class Demo:
    
    value = 11
    def __init__(self, A, B):
        
        self.No1 = A
        self.No2 = B
        
    def fun(self):
        print("Value of No1 is:", self.No1)
        print("Value of No2 is:", self.No2)
        print("Value of class variable is:", Demo.value)
        
    def gun(self):
        print("Value of No1 is:", self.No1)
        print("Value of No2 is:", self.No2)
        print("Value of class variable is:", Demo.value)
        
        
        
def main():
    
    obj1 = Demo(11, 21)
    obj2 = Demo(51, 101)

    obj1.fun()
    obj2.fun()
    
    obj1.gun()
    obj2.gun()
   
    
if __name__ =="__main__":
    main()    
