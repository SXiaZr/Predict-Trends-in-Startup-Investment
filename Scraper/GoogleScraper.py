# coding: utf-8
# -------------------- Package import -------------------- #
# Import parent package
from Scraper import ScraperBase
# requests and parser packages
import requests
from bs4 import BeautifulSoup
import urllib
# other packages
import re
import datetime
from collections import Counter
import warnings; warnings.filterwarnings('ignore') 

# -------------------- Google scraper class -------------------- #
class google_scraper(ScraperBase.scraper):
    """
    Google scraper, adapted from: https://github.com/meibenjin/GoogleSearchCrawler/blob/master/gsearch.py
    Date: 28-03-2019
    """
    def __init__(self, results_per_page=15):
        """
            Initialize the Scraper.
            Input:
                results_per_page:              Number of result per page
        """
        super(google_scraper, self).__init__()
        self.name = 'Google'
        self.host = 'www.google.com'
        self._results_per_page = results_per_page
        self.initialize_requests()
        
    def set_header(self):
        """
        Set the header of request.
        """
        self._header['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3'
        self._header['referer']= "https" + self.host
        self._session.headers  = self._header
    
    def _parse_results(self, r, query):
        """
        Return a list extract serach results list from downloaded html file.
       
        Input:
            r:                       Response object of request
            query:                   Keyword of query
        """
        results = []
        # Use 'BeautifulSoup' package to parse html
        soup = BeautifulSoup(r.text, 'html.parser')
        # Reset parse flag
        self._parseFlag = True
        try:
            div = soup.find('div', id='search')
            if (type(div) != type(None)):
                lis = div.findAll('div', {'class': 'g'})
                if (len(lis) > 0):
                    for li in lis:
                        # Extract titles
                        r = li.find('div', {'class': 'r'})
                        if (type(r) == type(None)):
                            continue
                        h3 = r.find('h3')
                        if (type(h3) != type(None)):
                            title = h3.text
                        else:
                            title = ''
                        # Extract time
                        span_st = li.find('span', {'class': 'st'})
                        div_slp = li.find('div', {'class': 'slp f'})
                        time_flag1 = (type(span_st) == type(None))
                        time_flag2 = (type(div_slp) == type(None))
                        if (time_flag1 and time_flag2):
                            continue
                        if not time_flag1:
                            content = span_st.text
                            content = re.sub(r'<.+?>', '', content)
                        if not time_flag2:
                            timeRex = re.compile('[A-Z][a-z]{1,} [0-9]{1,}, [0-9]{4,}', re.M)
                            time_raw = timeRex.findall(div_slp.text)
                            if (len(time_raw) > 0):
                                time = time_raw[0]
                            else:
                                time = ''
                        elif not time_flag1:
                            span_f = span_st.find('span', {'class': 'f'})
                            if (type(span_f) != type(None)):
                                time_raw = span_f.text
                                time = time_raw.strip(' -')
                            else:
                                time_raw = ''
                                time = ''
                            content = re.sub(time_raw, '', content)
                        # Only return time results
                        if time != '':
                            try:
                                time = datetime.datetime.strptime(time, '%b %d, %Y')
                                results.append(time)
                            except:
                                pass
        except:
            self._parseFlag = False
            print("Parsing google data of {} fails.".format(query))
            return None
        return results
    
    def _get_date(self, results):
        """
        According to the relevance and frequency of time of search result, get the year and month from parsed data.
        
        Input:
            results:                      Results of parsed data
        """
        year_score_dict = {}
        for date, score in zip(results, list(range(len(results)))[::-1]):
            year = date.year
            if year in year_score_dict:
                year_score_dict[year]['score'] += score
                year_score_dict[year]['date'].append(date)
            else:
                year_score_dict[year] = {}
                year_score_dict[year]['score'] = score
                year_score_dict[year]['date']  = []
                year_score_dict[year]['date'].append(date)
                
        # Get the year with highest weight
        if len(year_score_dict.items()) != 0:
            year = max(year_score_dict.items(), key=lambda x: x[1]['score'])[0]
        else:
            # If cannot get the time info, assign it to 2199 year, which can be filtered later
            return str(datetime.datetime(2199, 1, 1))
            # Get the most common month of this year
        month = Counter([i.month for i in year_score_dict[year]['date']]).most_common()[0][0]
        return str(datetime.datetime(year, month, 1))

    def search(self, query, lang='en', num=None):
        """
        Return a list of lists search web
        
        Input:
            query:                 Query key words
            lang:                  Language of search results
            num:                   Number of search results to return
        """
        search_results = []
        results = []
        # Quoting HTML form values when building up a query string to go into a URL
        query_decode = urllib.parse.quote(query)
        # Compute pages needed to search
        if num is None:
            num = self._results_per_page
        if (num % self._results_per_page == 0):
            pages = int(num / self._results_per_page)
        else:
            pages = num // self._results_per_page + 1
            
        # Form the url of request
        for p in range(0, pages):
            start = p * self._results_per_page
            url = '%s/search?hl=%s&num=%d&start=%s&as_q=%s' % ("https://" + self.host, 
                                                               lang, 
                                                               self._results_per_page, 
                                                               start, 
                                                               query_decode)
            response = self._session.get(url)
            parsed_data = self._parse_results(response, query)
            if self._parseFlag:
                results.extend(parsed_data)
        return {'title': query, 'InterviewTimeAuto': self._get_date(results)}