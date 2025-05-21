
def Display(Num):
    for i in range(Num):
        for j in range(i,Num):
            print("* ",end = " ")
        print() 
           
def main():
    No = 0
    print("Enter first number: ")
    No = int(input())
    Display(No)
     
    
if __name__ =="__main__":
    main()