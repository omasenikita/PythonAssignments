class BankAccount:
    ROI = 10.5
    
    def __init__(self, A, B,):
        self.Name = A
        self.Amount = B
        
    def Deposit(self):
        print(self.Name, "Welcome to the bank")
        print("Your current balance is : ", self.Amount)
        num = 0
        print("Enter the amount you want to deposit")
        num = int(input())
       
        self.Amount = self.Amount+num
        
        
    def Withdraw(self):
        print("Your current balance is : ", self.Amount)
        num = 0
        print("Enter the amount you want to withdraw :")
        num = int(input())
        self.Amount = self.Amount - num
        
        
        
    def CalculateInterest(self):
        interest = self.Amount * BankAccount.ROI / 100
        return interest
    
    def Display(self):
        print("Name of the account holder is : ", self.Name)
        print("Amount in the account is : ", self.Amount)
        print("Rate of interest is : ", BankAccount.ROI)
        print("Interest on the amount is : ", self.CalculateInterest())
        
        
def main():
  
    obj = BankAccount("Nikita",9000)
    
    obj.Deposit()  
    obj.Withdraw()  
    obj.CalculateInterest()
    obj.Display()
    
    obj1 = BankAccount("Kalyani",10000)
    
    obj1.Deposit()  
    obj1.Withdraw()  
    obj1.CalculateInterest()
    obj1.Display()
    
    
if __name__ == "__main__":
    main()