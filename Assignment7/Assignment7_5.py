def ChkPallindrome(str1):
    rev = ""
    for ch in str1:
        rev = ch + rev
    print(rev)    

    if (str1 == rev):
        print("String is pallindrom")
    else:
        print("string is not Pallindrome")    
            
        
        
        
        
def main():
    string = ""
    
    print("Enter a string")
    string = input()
    
    ChkPallindrome(string)
    
    
    
     
           
if __name__ == "__main__":
    main()