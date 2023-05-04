from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException,TimeoutException,StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your program')

# Add an argument for the first string
parser.add_argument('string1', type=str, help='Description of the first string')

# Add an argument for the second string
parser.add_argument('string2', type=str, help='Description of the second string')

# Add an argument for the second string
parser.add_argument('string3', type=str, help='Description of the third string')

# Add an argument for the second string
parser.add_argument('string4', type=str, help='Description of the third string')

# Parse the command-line arguments
args = parser.parse_args()

URL=args.string1
output_path = args.string2
mode= args.string3
#0: past data
#1: current matches
#2: last week match
#3: every upcoming match
#4: specific week
extra= args.string4

def extract_data(driver, csvwriter):
    try:
        table_container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "table-container")))
    except TimeoutException:
        print("Timed out waiting for table container to be present")
        return

    try:
        matches_table = WebDriverWait(table_container, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "matches")))
    except TimeoutException:
        print("Timed out waiting for matches table to be present")
        return

    try:
        tbody = WebDriverWait(matches_table, 10).until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
    except TimeoutException:
        print("Timed out waiting for tbody to be present")
        return

    try:
        rows = WebDriverWait(tbody, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "tr")))
    except TimeoutException:
        print("Timed out waiting for rows to be present")
        return
    date =""
    retries = 3
    for row in rows:
        while True:
            try:
                if "no-date-repetition-new live-now" in row.get_attribute("class"):
                    break
                if "border border betting" in row.get_attribute("class"):
                    break
                if "no-date-repetition-new" in row.get_attribute("class"):
                    date = row.find_element(by=By.CLASS_NAME, value="date").text.split(" ")[1]
                    break
                home_team = row.find_element(by=By.CSS_SELECTOR, value=".team.team-a").text

                try:
                    away_team = row.find_element(by=By.CSS_SELECTOR, value=".team.team-b").text
                except NoSuchElementException:
                    away_team = "N/A"

                score = row.find_element(by=By.CSS_SELECTOR, value=".score-time").text
                hg, ag = score.split("-")
                hg =hg.strip()
                ag = ag.strip()
                res=""
                if(hg>ag):
                    res='H'
                elif(ag>hg):
                    res='A'
                else:
                    res='D'

                csvwriter.writerow([date,home_team, away_team, hg, ag,res])
                break
            except StaleElementReferenceException:
                retries -= 1
                if retries == 0:
                    raise
                time.sleep(3)
# Set up the driver
driver = webdriver.Chrome()

# Load the webpage
driver.get(URL)

# Open a CSV file for writing with 'utf-8' encoding
with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Date","Home", "Away", "HG", "AG","Res"])
    match mode:
        case "0":
            # Loop until the "Previous" link is disabled
            while True:
                # Check if the "Previous" link is enabled
                previous_link = driver.find_element(by=By.ID,value="page_competition_1_block_competition_matches_summary_9_previous")
                previous_link.click()
                time.sleep(2)
                # Extract the data from the current page
                extract_data(driver, csvwriter)
                

                if "disabled" in previous_link.get_attribute("class"):
                    # If the link is disabled, break out of the loop
                    break
                else:
                    # If the link is enabled, click it and wait for the page to load
                    previous_link.click()
                    time.sleep(2)
                    driver.implicitly_wait(100)
        case "1":
            extract_data(driver, csvwriter)
        case "2":
            previous_link = driver.find_element(by=By.ID,value="page_competition_1_block_competition_matches_summary_9_previous")
            previous_link.click()
            time.sleep(2)
            extract_data(driver, csvwriter)
        case "3":
                        # Loop until the "Previous" link is disabled
            while True:
                # Check if the "Previous" link is enabled
                next_link = driver.find_element(by=By.ID,value="page_competition_1_block_competition_matches_summary_9_next")
                next_link.click()
                time.sleep(2)
                # Extract the data from the current page
                extract_data(driver, csvwriter)
                

                if "disabled" in next_link.get_attribute("class"):
                    # If the link is disabled, break out of the loop
                    break
                else:
                    # If the link is enabled, click it and wait for the page to load
                    next_link.click()
                    time.sleep(2)
                    driver.implicitly_wait(100)
        case "4":
            print('NOT YET FINISHED')
        case _:
            csvwriter.writerow("161")
        
# Quit the driver
driver.quit()
