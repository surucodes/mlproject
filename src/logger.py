# import logging 
# import os 
# from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
# os.makedirs(logs_path,exist_ok =True)

# LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# logging.basicConfig(
#     filename= LOG_FILE_PATH,
#     format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level= logging.INFO,

# )

import logging
import os
from datetime import datetime

# Create a timestamped log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory path (only the directory, not the file)
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create the full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
# This is configuring the default behavior of the logger — like telling it where and how to store logs.

#this is a function(basicConfig) call from the module logging.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    #format of the logs that are going to be saved in my log file who's name is going to be like logs/timestamp.logs
    level=logging.INFO,
#It means:
# Log everything that’s INFO, WARNING, ERROR, or CRITICAL
# But ignore DEBUG messages
)
#we log in the basic info eg the data ingestion is successful and the time stamp will be the files name itself 
# we log in errors as well eg this exception occured here at this time etc
# import logging
# logging.debug("This is just a debug message")
# logging.info("Program started")
# logging.warning("Disk space getting low")
# logging.error("Failed to open the data file")
# logging.critical("System crash! Immediate action required!")


#test
#if __name__ == "__main__":
#    logging.info("Logging has started")