from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException,TimeoutException,StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

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
driver.get("https://int.soccerway.com/national/japan/j1-league/2023/regular-season/r73435/")

# Open a CSV file for writing with 'utf-8' encoding
with open("JPN.csv", "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Date","Home", "Away", "HG", "AG","Res"])
    
    # Loop until the "Previous" link is disabled
    while True:
        # Extract the data from the current page
        extract_data(driver, csvwriter)
        
        # Check if the "Previous" link is enabled
        previous_link = driver.find_element(by=By.ID,value="page_competition_1_block_competition_matches_summary_9_previous")
        if "disabled" in previous_link.get_attribute("class"):
            # If the link is disabled, break out of the loop
            break
        else:
            # If the link is enabled, click it and wait for the page to load
            previous_link.click()
            time.sleep(1)
            driver.implicitly_wait(100)
        
# Quit the driver
driver.quit()