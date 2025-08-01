import time
import os
from datetime import datetime
import schedule

def log_time_to_file(filename="timestamp_log.txt"):
    
    """
    Logs the current timestamp to a file every `interval_seconds`.
    Creates a new file if it doesn't exist, otherwise appends to it.
    """
    
    print("Logging timestamps to  5 seconds. Press Ctrl+C to stop.")
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if the file exists
            if not os.path.exists(filename):
                # Create the file in write mode ('w') if it doesn't exist
                f = open(filename, 'w')
                f.write(f"Log started at: {current_time}\n")
                print(f"Created new file: {filename}")
            
            # Open the file in append mode ('a') to add new lines
            f = open(filename, 'a')
            f.write(f"{current_time}\n")
            
            time.sleep(5)
            print(f"Written: {current_time}")
           
            f.close()
            
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")

def main():
    print("Enter the filename to log timestamps (default is 'timestamp_log.txt'):")
    filename = input().strip() 
    if not filename:
        filename = "timestamp_log.txt"
    # Schedule the log_time_to_file function to run every 5 seconds
    schedule.every().seconds.do(log_time_to_file, filename=filename)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

    
if __name__ == "__main__":
    main()
   