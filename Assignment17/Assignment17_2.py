
import schedule
import time
import datetime 


def Display():
    current_time = datetime.datetime.now()
    print("Current time is:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
   
def main():
    # Schedule the Print function to run every 5 seconds
    schedule.every(1).minutes.do(Display)

    while True:
        # Run pending tasks
        schedule.run_pending()
        time.sleep(1)              # Sleep for a short duration to avoid busy waiting
    
if __name__ == "__main__":
    main()