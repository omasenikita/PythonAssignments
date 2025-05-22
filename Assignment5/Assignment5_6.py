def RectAreaPeri(len, wid):
    Area = len * wid
    print("Area of Rectangle is :",Area)
    
    Perim = 2 * (len + wid)
    print("Perimeter of reactangle is : ",Perim)
    
   
def main():
    Length = Width = 0
    print("Enter a length of rectangle : ")
    Length = int(input())
    print("Enter a width of rectangle : ")
    Width = int(input())
    
    RectAreaPeri(Length, Width)
      
if __name__ == "__main__":
    main()
