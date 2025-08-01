
import schedule
import time

def Print():
    print("Lunch time ")
    
def Print2():
    print("Wrap up your work ")

def main():
    schedule.every().day.at("13:00").do(Print)
    schedule.every().day.at("18:00").do(Print2)

    
    while True:
        # Run pending tasks
        schedule.run_pending()
        time.sleep(1)              # Sleep for a short duration to avoid busy waiting
    
if __name__ == "__main__":
    main()