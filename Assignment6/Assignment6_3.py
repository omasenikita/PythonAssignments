def Table(num):
    iCnt = 1
    while(iCnt <= 10):
        
        result = num * iCnt
        print(num,"X",iCnt,"=",result)
        iCnt = iCnt + 1

   
    
def main():
    Value = 0 
    print("Enter a number : ")
    Value = int(input())
    Table(Value)
     
if __name__ == "__main__":
    main()
