def main():
    
    fname = open("Student.txt","w")
    print("How many names you wnat to enter ?")
    name = int(input())
    for i in range(name):
        data = input()
        fname.write(data+"\n")
    
if __name__ == "__main__":
    main()