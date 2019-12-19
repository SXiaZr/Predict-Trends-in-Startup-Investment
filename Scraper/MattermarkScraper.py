# coding: utf-8
# -------------------- Package import -------------------- #
# requests and parser packages
import requests
from bs4 import BeautifulSoup
# selenium package
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# other packages
import re
import math
import warnings; warnings.filterwarnings('ignore')

# -------------------- mattermark scraper class -------------------- #
class mattermark_scraper():
    """
    Scraper for Mattermark. (Version: 22-Feb-2019)
    """
    # -------------------- Private functions -------------------- #
    def __init__(self, username, password):
        """
        Initialize the Scraper.
        """
        self.username = username
        self.investor_data = []
        self._login_url = 'https://mattermark.com/app/'
        self.__driver = webdriver.Chrome()
        self.__CookieJar = requests.cookies.RequestsCookieJar()
        self.__password = password
        
    def __change_proxy_selenium(self):
        """
        Change proxy of brower when being blocked. Better to have a proxy list.
        """
        PROXY = "35.162.25.177:3128" # IP:PORT or HOST:PORT
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--proxy-server=http://%s' % PROXY)
        self.__driver = webdriver.Chrome(chrome_options=chrome_options)
    
    def __send_cookies_to_requests(self):
        """
        Get cookies after login for requests.
        """
        cookies = self.__driver.get_cookies()
        for i in cookies:    # Add cookie to CookieJar
            self.__CookieJar.set(i["name"], i["value"])
            
    def __parse_investor_data(self, r):
        """
        Parse investor data. We abtain a json, and keep all items.
        """
        return r.json()['results']
        
    def __get_authentication(self):
        """
        Get authentication token.
        """
        header = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                  'Host': 'mattermark.com',
                  'Upgrade-Insecure-Requests': '1',
                  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36'}
        r = requests.get('https://mattermark.com/app/investors', 
                         headers=header, 
                         cookies=self.__CookieJar, 
                         verify=False)
        api_authentication_token = re.search('var token = \"(.*)\"', r.text, re.M)
        return api_authentication_token[1]
    
    def __get_page_number(self, header):
        """
        Page number is form first requery.
        """
        r = self.__get_one_page_investor_data(header, 1)
        return r.json()['meta']['total_pages']
    
    def __get_one_page_investor_data(self, header, page_index):
        """
        Get one page investor data of page_index.
        """
        # Form posted data
        data = {"dataset":"investor",
                "sort":[{"three_year_funds_sold": "desc"}],
                "page":page_index,"per_page":50,
                "filter":{"type":{"in":["vc","angel","accelerator"]}}}
        data = re.sub('\'', '\"', str(data))
        # Post
        r = requests.post('https://api.mattermark.com/queries', 
                          headers=header, 
                          data=data, 
                          cookies=self.__CookieJar, 
                          verify=False)
        return r
        
    # -------------------- Public functions -------------------- #        
    def login(self):
        """
        Use selenium to automatically login.
        """
        # Open login url
        self.__driver.get(self._login_url)
        # Input username and passward
        self.__driver.find_element_by_xpath("//input[@name='email']").send_keys(self.username)
        self.__driver.find_element_by_xpath("//input[@name='password']").send_keys(self.__password)
        # Click 'Sign in' button 
        self.__driver.find_element_by_xpath("//input[@id='signin']").click()
        # Wait until response
        WebDriverWait(self.__driver, 20) \
            .until(EC.presence_of_element_located((By.XPATH, "//title[text()='Mattermark - Quantifying Private Company Growth']")))
        
    def get_cookies(self):
        '''
            Access to cookies.
        '''
        cookies = self.__driver.get_cookies()
        return cookies
    
    def scrape_investor_data(self):
        """
        Scrape investor data from 'Research -> Investors'.
        """
        # Get cookies
        self.__send_cookies_to_requests() 
        # Get authentication token
        api_authentication_token = self.__get_authentication()
        # Form a new header
        header = {'Content-Type': 'application/json',
                  'Host': 'api.mattermark.com',
                  'Origin': 'https://mattermark.com',
                  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36'}
        header['Authentication'] = api_authentication_token
        # Loop to scrape each page
        page_number = self.__get_page_number(header)
        basis = math.ceil(page_number / 100)
        for page in range(1, page_number+1):
            temp = self.__get_one_page_investor_data(header, int(page))
            temp = self.__parse_investor_data(temp)
            self.investor_data += temp
            # Report the precedure
            if page % basis == 0:
                percentage = 100 * page / page_number
                print(f'{percentage:.2f}% tasks have been finished.', end = '\r')
            # time sleep to avoid detection
            # time.sleep(1)
        print(f'The scraping task havs been finished!')