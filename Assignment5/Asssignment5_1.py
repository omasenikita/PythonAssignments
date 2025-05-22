def ChkVowel(char):
    if((char == 'a')or(char == 'e')or(char == 'i')or(char == 'o')or(char == 'u')):
        print("Charater is vowel ")
    else:
        print("Character is Consonant")
    
    
def main():
    print("Enter the character : ")
    ch = input()
    
    ChkVowel(ch)
   
    
    
if __name__ == "__main__":
    main()
