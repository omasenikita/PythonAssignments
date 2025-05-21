
def Display(Num):
    
    for i in range(1,Num+1):
        for j in range(1,i+1):
             print(j,end = " ")
        print() 
           
def main():
    No = 0
    print("Enter first number: ")
    No = int(input())
    Display(No)
     
    
if __name__ =="__main__":
    main()