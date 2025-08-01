
import schedule
import time
def Print():
    print("Do Coding")

def main():
    # Schedule the Print function to run every 5 seconds
    schedule.every(30).minutes.do(Print)

    while True:
        # Run pending tasks
        schedule.run_pending()
        time.sleep(1)              # Sleep for a short duration to avoid busy waiting
    
if __name__ == "__main__":
    main()