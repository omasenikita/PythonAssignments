import multiprocessing

def Factorial(no):
    fact = 1
    for i in range(1, no + 1):
        fact *= i
    return fact

def main():
    l1 = []
    print("Enter how many numbers you want to calculate the factorial of:")
    num = int(input())

    print(f"Enter {num} numbers:")
    for _ in range(num):
        no = int(input())
        l1.append(no)

    pool = multiprocessing.Pool() 
    result = pool.map(Factorial, l1)

    pool.close()  
    pool.join()

    print("Factorials of the numbers:")
    for i in range(num):
        print(f"{l1[i]}! = {result[i]}")

if __name__ == "__main__":
    main()
