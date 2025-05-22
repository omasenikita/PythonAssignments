Cube = lambda x : x ** 3
square = lambda x : x * x
   
def main():
    num = 0
    print("Enter a number")
    num = int(input())
    
    iRet1= Cube(num)
    print("cube is",iRet1)
    
    iRet2 = square(num)
    print("square is",iRet2)

if __name__ == "__main__":
    main()
