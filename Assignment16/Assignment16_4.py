def main():
    print("How many names you wnat to enter ?")
    fname = open("Number.txt","w")

    name = int(input())
    for i in range(name):
        data = int(input())
        fname.write(str(data)+"\n")
        
            
    
if __name__ == "__main__":
    main()