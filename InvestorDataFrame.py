# coding: utf-8
# -------------------- Package import -------------------- #
# Import scraper packages
from Scraper import youtube_scraper, crunchbase_scraper, google_scraper
# Import other packages
import os
import re
import math
import numpy as np
import pandas as pd
DATA_PATH = './data/'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# -------------------- InvestorDataFrame class -------------------- #
class InvestorDataFrame():
    """
    A class to gather and clean all the data about an investor into a DataFrame.
    """
    def __init__(self, investor, youtube_data_num=30):
        """
        Class initializer.
        
        Input:
            investor:               Name of the investor
            youtube_data_num:       Number of youtube data to scrape 
        """        
        self._investor = investor
        self._pipeline_flag  = None
        self._youtube_df     = pd.DataFrame(columns=['publishedAt', 'title', 'videoId','caption_text'])
        self._crunchbase_df  = pd.DataFrame(columns=['Domain', 'Money_Raised', 'Announced_Date'])
        self._real_time_auto = pd.DataFrame(columns=['title', 'InterviewTimeAuto'])
        # Precess
        self._pipeline_flag = self.pipeline(youtube_data_num)
        
        
    def get_youtube_data(self, data_num=30):
        """
        Get subtitles data of investor from youtube and save it to local in '.csv' format.
        Structured the data into DataFrame as "._youtube_df". Default get subtitles of 30 investors.
        
        Input:
            data_num:               Number of youtube data to scrape
        Output:
            Flag:                   Whether the Load the aimed data
        """
        print("---------- Load youtube Data ----------")
        file = DATA_PATH + self._investor.replace(' ', '_') + '_youtube.csv'
        if os.path.exists(file):
            investor_df = pd.read_csv(file) 

        else:
            youtube_data_list = []
            temp = youtube_scraper(self._investor)
            temp.scrape_interview_data(data_num)

            # Only keep videos with English subtitles
            englishSign = "[A-Za-z0-9\s\.,-\/#!$%\^&\*;:{}=\-_`~()@\+\?><\[\]\+\'\"]*"
            regulerEx   = re.compile(englishSign)
            for i in temp.caption_data[self._investor]: # Need to adjust
                letter_num = np.sum([len(t) for t in regulerEx.findall(i['caption_text'])])
                total_num  = len(i['caption_text'])
                # Filiter video: have caption in other language; have no caption; have white space caption 
                if letter_num / total_num > 0.8 and len(i['caption_text']) > 100 \
                                                and len(i['caption_text'].replace(" ", '')) > 100:
                    youtube_data_list.append({'title'       : i['title'],
                                              'videoId'     : i['videoId'],
                                              'publishedAt' : i['publishedAt'],
                                              'caption_text': i['caption_text']})
            
            # Append new data to DataFrame and save to local
            if len(youtube_data_list) > 0:
                investor_df = pd.DataFrame.from_dict(youtube_data_list)
                investor_df.drop_duplicates('title', inplace=True)  
                investor_df.to_csv(file)
            else:
                print('{}: Youtube data is not available.'.format(self._investor))
                return False
               
        self._youtube_df = pd.concat([self._youtube_df, investor_df], join='inner')
        self._youtube_df.drop_duplicates('title', inplace=True) 
        print('{}: Youtube data has been loaded.'.format(self._investor))
        return True
        
    def get_crunchbase_data(self):
        """
        From crunchbase, get investment data of investor and save it to local in '.csv' format.
        Structured the data into DataFrame as "._crunchbase_df".
        """
        print("---------- Load Crunchbase Data ----------")
        file = DATA_PATH + self._investor.replace(' ', '_') + '_crunchbase.csv'
        if os.path.exists(file):
            investor_df = pd.read_csv(file)
            
        else:  
            crunchbase_data_list = []
            # Get categories of the latest 10 companies the investor invested
            c_scraper = crunchbase_scraper()
            investor_data = c_scraper.scrape_investor_data(self._investor)
            # If we cannot find the investor in crunch base
            if not c_scraper._linkFlag or not c_scraper._parseFlag:
                print('{}: Crunchbase data is not available.'.format(self._investor))
                return False
            else:
                investor_data = pd.DataFrame.from_dict(investor_data, orient='index')
            
            # Gather the name of invested companies
            invest_company_list = []
            page_number = investor_data.shape[0]
            page, basis = 0, math.ceil(page_number / 100)
            for _, values in investor_data.iterrows():
                company_name_in_url = values['Ref']
                company_name_in_url = re.search("/organization/(.*)", company_name_in_url, re.M)[1]
                company_data = c_scraper.scrape_company_data(company_name_in_url)
                if c_scraper._linkFlag and c_scraper._parseFlag:
                    date_time    = pd.to_datetime(values['Announced_Date'])
                    for i in company_data['Basic_Info']['Categories']:
                        crunchbase_data_list.append({'Domain'        : i, 
                                                     'Announced_Date': date_time,
                                                     'Money_Raised'  : values['Money_Raised']})
                page += 1
                # Report the precedure
                if page % basis == 0:
                    percentage = 100 * page / page_number
                    print(f'{percentage:.2f}% scraping task has been finished.', end = '\r')
            print('The scraping task has been finished!\t\t\t')
            
            # Append new data to DataFrame and save to local
            if len(crunchbase_data_list) > 0:
                investor_df = pd.DataFrame.from_dict(crunchbase_data_list)
                investor_df.to_csv(file)
            else:
                print('{}: Crunchbase data is not available.'.format(self._investor))
                return False

        investor_df['Announced_Date'] = investor_df['Announced_Date'].apply(lambda x: pd.to_datetime(x))
        self._crunchbase_df = pd.concat([self._crunchbase_df, investor_df], join='inner')
        self._crunchbase_df.drop_duplicates(inplace=True) 
        print('{}: Crunchbase data has been loaded.'.format(self._investor))
        return True
                    
    def get_published_date(self):
        """
        Automatically get published date from results of google search.
        """
        print("---------- Load PublishedDate Data ----------")
        file = DATA_PATH + self._investor.replace(' ', '_') + '_pubdate.csv'
        if os.path.exists(file):
            investor_df = pd.read_csv(file) 
            
        else:  
            pub_time_list = []
            g_scraper = google_scraper()
            page_number = self._youtube_df.shape[0]
            page, basis = 0, math.ceil(page_number / 100)
            for i, series in self._youtube_df.iterrows():
                pub_time_list.append(g_scraper.search(series['title'], lang='en')) 
                page += 1
                # Report the precedure
                if page % basis == 0:
                    percentage = 100 * page / page_number
                    print(f'{percentage:.2f}% scraping task has been finished.', end = '\r')
            print('The scraping task has been finished!\t\t\t')
            
            # Append new data to DataFrame and save to local
            if len(pub_time_list) > 0:
                investor_df = pd.DataFrame.from_dict(pub_time_list)
                investor_df.to_csv(file)
            else:
                print('{}: Published date data is not available.'.format(self._investor))
                return False
            
        investor_df['InterviewTimeAuto'] = investor_df['InterviewTimeAuto'].apply(lambda x: pd.to_datetime(x))
        self._real_time_auto = pd.concat([self._real_time_auto, investor_df], join='inner')
        self._real_time_auto = self._real_time_auto.drop_duplicates('title')
        print('{}: Published date data has been loaded.'.format(self._investor))
        return True     

    def pipeline(self, youtube_data_num=30):
        """
        An entire pipeline of getting data.
        
        Input:
            youtube_data_num:           Number of youtube data to scrape  
        """
        print("-------------------- {} --------------------".format(self._investor))
        pipelineList = [self.get_crunchbase_data, 
                        self.get_youtube_data, 
                        self.get_published_date]
  
        for func in pipelineList:
            if not func():
                print("---------- All Scraping Data ----------")  
                print('{}: All scraped data loading...Fail;('.format(self._investor))
                return False
        print("---------- All Scraping Data ----------")  
        print('{}: All scraped data loading...Success:)'.format(self._investor))
        print("---------------------{}---------------------".format('-'*len(self._investor)))
        return True