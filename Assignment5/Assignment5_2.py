def ChkAge(Age):
    if(Age >= 18):
        print("You can vote")
    else:
        print("You cannot vote")
    
    
def main():
    Age = 0
    print("Enter Your age : ")
    Age = int(input())
    
    ChkAge(Age)
   
    
    
if __name__ == "__main__":
    main()
