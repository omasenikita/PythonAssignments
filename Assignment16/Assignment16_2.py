def main():
    
    fname = open("data.txt","a")
    print("How many names you wnat to enter ?")
    name = int(input())
    for i in range(name):
        data = input()
        fname.write(data+"\n")
    fname =open("data.txt","r")
        
    dataX= fname.read()
    print(dataX)
    
if __name__ == "__main__":
    main()