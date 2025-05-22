def ChkNumber(num):
   
    if((num % 2 )== 0):
        print("Number is Even")
    else:
        print("Number is Odd")    
                   
def main():
   Num = 0
   print("Enter a number")
   Num = int(input())
   
   ChkNumber(Num)
      
if __name__ == "__main__":
    main()
