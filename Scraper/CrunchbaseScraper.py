# coding: utf-8
# -------------------- Package import -------------------- #
# Import parent package
from Scraper import ScraperBase
# requests and parser packages
import requests
from bs4 import BeautifulSoup
# other packages
import re
import warnings; warnings.filterwarnings('ignore')

# -------------------- crunchbase scraper class -------------------- #
class crunchbase_scraper(ScraperBase.scraper):
    """
    Scraper for Crunchbase. (Version: 24-Feb-2019)
    """
    def __init__(self):
        """
        Initialize the Scraper. 
        """
        super(crunchbase_scraper, self).__init__()
        self.name = 'Crunchbase'
        self.host = 'www.crunchbase.com'
        self.investor_data = {}
        self.company_data  = {}
        self.initialize_requests()
        
    def _parse_investor_data(self, r, investor):
        """
        Parse investor data of 'https://www.crunchbase.com/person/investor_name'.

        Input:
            r:                       Response object of request
            investor:                Name of the investor
        """
        agg_data = {}
        # Use 'BeautifulSoup' package to parse html
        soup = BeautifulSoup(r.text, 'html.parser')
        # Reset parse flag
        self._parseFlag = True
        # Parse the content of html, return False if any bad parsing
        try:
            investment_raw_table     = soup.find('table', {'class':"card-grid ng-star-inserted"})
            investment_raw_data_list = investment_raw_table.find_all('tr',{'class':"ng-star-inserted"})
            for investment_raw_data in investment_raw_data_list:
                investment_temp   = investment_raw_data.find_all('div',\
                                        {'class':"flex-no-grow cb-overflow-ellipsis identifier-label"})
                Announced_Date    = investment_raw_data.find('span',\
                                        {'class':"component--field-formatter field-type-date ng-star-inserted"}).text
                Organization_Name = investment_temp[0].text.strip()
                Lead_Investor     = investment_raw_data.find('span',\
                                        {'class':"component--field-formatter field-type-boolean ng-star-inserted"}).text 
                Funding_Round     = investment_temp[1].text.strip()
                Money_Raised      = investment_raw_data.find('span',\
                                        {'class':"component--field-formatter field-type-money ng-star-inserted"}).text
                Ref               = investment_raw_data.find('a', \
                                        {'class':"cb-link component--field-formatter field-type-identifier ng-star-inserted"})['href']
                agg_data[Organization_Name] = {'Announced_Date': Announced_Date, 
                                               'Lead_Investor' : Lead_Investor, 
                                               'Funding_Round' : Funding_Round,
                                               'Money_Raised'  : Money_Raised,
                                               'Ref'           : Ref}
        except:
            self._parseFlag = False
            print("Parsing investor data of {} fails.".format(investor))
            return None
        # Return True and parsed data if it's well-parsed
        return agg_data
    
    def _parse_company_data(self, r, company):
        """
        Parse company data of 'https://www.crunchbase.com/organization/company_name'.
        
        Input:
            r:                       Response object of request
            company:                 Name of the company
        """
        soup = BeautifulSoup(r.text, 'html.parser')
        # Reset parse flag
        self._parseFlag = True
        
        try:
            # Scrape introduction data
            keywords_raw_list     = soup.find_all('span', {'class':"component--field-formatter field-type-identifier-multi"})
            Companies_Location    = keywords_raw_list[0].find_all('a', {'class':"cb-link ng-star-inserted"})
            Companies_Location    = [i.text.strip() for i in Companies_Location]
            Categories            = keywords_raw_list[1].find_all('a', {'class':"cb-link ng-star-inserted"})
            Categories            = [i.text.strip() for i in Categories]
            Headquarters_Regions  = keywords_raw_list[2].find_all('a', {'class':"cb-link ng-star-inserted"})
            Headquarters_Regions  = [i.text.strip() for i in Headquarters_Regions]
            # Aggregate introduction data
            agg_data_intro        = {'Companies_Location'  : Companies_Location, 
                                     'Categories'          : Categories,
                                     'Headquarters_Regions': Headquarters_Regions}

            # Scrape table content
            company_data_temp = []
            self.find_element      = lambda i, tag, attribute, value: i.find(tag, {attribute: value}).text.strip() if type(i.find(tag, {attribute: value})) != type(None) else None
            Funding_Rounds_Section = soup.find('section-layout', {'cbtableofcontentsitem':"Funding Rounds"})
            Total_Funding_Amount   = Funding_Rounds_Section.find('a', 
                                       {'class':"cb-link component--field-formatter field-type-money ng-star-inserted"}).text
            table_section          = Funding_Rounds_Section.find('tbody')
            table_list             = table_section.find_all('tr', {'class':"ng-star-inserted"})
            for i in table_list:
                Announced_Date      = self.find_element(i, 'span','class',
                                                        'component--field-formatter field-type-date ng-star-inserted')
                Transaction_Name    = self.find_element(i,'div','class',
                                                        'flex-no-grow cb-overflow-ellipsis identifier-label')
                Number_of_Investors = self.find_element(i, 'a', 'class',
                                                        'cb-link component--field-formatter field-type-integer ng-star-inserted')
                Money_Raised        = self.find_element(i, 'span', 'class',
                                                        'component--field-formatter field-type-money ng-star-inserted')
                Lead_Investors      = self.find_element(i, 'a', 'class', 
                                                        'cb-link ng-star-inserted')
                agg_data_investment = {'Announced_Date'     : Announced_Date,
                                       'Transaction_Name'   : Transaction_Name,
                                       'Number_of_Investors': Number_of_Investors,
                                       'Money_Raised'       : Money_Raised,
                                       'Lead_Investors'     : Lead_Investors}
                company_data_temp.append(agg_data_investment)

            # Put all data into a dict
            company_data_return = {'Basic_Info'   :agg_data_intro, 
                                   'Investment'   :company_data_temp, 
                                   'Total_Funding':Total_Funding_Amount}
        except:
            self._parseFlag = False
            print("Parsing company data of {} fails.".format(company))
            return None
        return company_data_return
    
    def set_header(self):
        """
        Set the header of request.
        """
        self._header['Accept']        = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
        self._header['Cache-Control'] = 'max-age=0'
        self._header['Host']          = 'www.crunchbase.com'
        self._header['Upgrade-Insecure-Requests'] = '1'
        self._session.headers = self._header
        
    def scrape_investor_data(self, investor):
        """
        Scrape investment data of a specific investor.
        Input:
            investor:               Name of the investor
        """
        # Modify investor name
        investor_ori = investor
        investor = investor.lower()
        investor = re.sub('\s+', '-', investor)
        r = self._session.get('https://www.crunchbase.com/person/' + investor, verify=False)
        # Reset the link flag
        self._linkFlag = True
        # If can't not find investor in Crunchbase
        if r.status_code != 200:
            print('Investor "{}" cannot be find in Crunchbase'.format(investor_ori))
            self._linkFlag = False
            return None
        else:
            return_data = self._parse_investor_data(r, investor_ori)
            self.investor_data[investor] = return_data
            return return_data
        
    def scrape_company_data(self, company):
        """
        Scrape investment data of a specific company.
        Input:
            company:                Name of the company
        """ 
        # Modify compandy name
        company_ori = company
        company = company.lower()
        company = re.sub('\s+', '-', company)
        r = self._session.get('https://www.crunchbase.com/organization/' + company, verify=False)
        # Reset the link flag
        self._linkFlag = True
        # If can't not find investor in Crunchbase
        if r.status_code != 200:
            print('Company "{}" cannot be find in Crunchbase'.format(company_ori))
            self._linkFlag = False
            return None    
        else:
            return_data = self._parse_company_data(r, company_ori)
            self.company_data[company] = return_data
            return return_data