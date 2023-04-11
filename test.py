import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
# Specify the path to the ChromeDriver executable
chromedriver_path = "path/to/chromedriver"

# Set up the Chrome driver
driver = webdriver.Chrome(executable_path=chromedriver_path)

# Load the webpage
url = "https://livescore.football-data.co.uk/"
driver.get(url)

# Find the table containing the "England Championship" league
table = None
try:
    wait = WebDriverWait(driver, 10)
    table = wait.until(EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'BZcol-12')][contains(.//td, 'England Championship')]")))
except TimeoutException:
    print("Table not found")

# Define a regular expression pattern to match team names with "City", "FC", or "Utd"
pattern = re.compile(r"^(.*?)(\s+(City|FC|Utd|Athletic))?$", re.IGNORECASE)

# Extract the home and away team names and write to a CSV file
if table:
    rows = table.find_elements(By.TAG_NAME, "tr")
    with open("test.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Home", "Away"])
        for row in rows[1:]:
            home_team = row.find_elements(By.CSS_SELECTOR, ".BZcol-3.BZhomeTeam")
            away_team = row.find_elements(By.CSS_SELECTOR, ".BZcol-3.BZawayTeam")
            if home_team and away_team:
                home_name = pattern.sub(r"\1", home_team[0].text.strip())
                away_name = pattern.sub(r"\1", away_team[0].text.strip())
                writer.writerow([home_name, away_name])
# Close the browser
driver.quit()