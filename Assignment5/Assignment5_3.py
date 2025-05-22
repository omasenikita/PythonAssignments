def FindLargest(a, b , c):
    
    if((a > b) and (a > c)):
        print("A is largest number is: ",a)
        
    elif((b > a) and(b > c)): 
        print("A largest number is : ",b)
        
    else:
        print("A largest number is ",c)    
           
def main():
   Num1 =Num2 = Num3 = 0
   print("Enter first number")
   Num1 = int(input())
   print("Enter Second number")
   Num2 = int(input())
   print("Enter Third number")
   Num3 = int(input())
   FindLargest(Num1, Num2, Num3)
      
if __name__ == "__main__":
    main()
