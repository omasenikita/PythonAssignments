def Convert(celsius):
    result =  (celsius *  9 / 5) + 32
    return result
   
   
def main():
   Temp = 0 
   fRet = 0.0
   print("Enter a Temperature in celsius")
   Temp= int(input())

   fRet = Convert(Temp)
   print("Temprature in fahenheit",fRet,"F")
      
if __name__ == "__main__":
    main()
