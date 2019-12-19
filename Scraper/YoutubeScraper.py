# coding: utf-8
# -------------------- Package import -------------------- #
# Import parent package
from Scraper import ScraperBase
# requests and parser packages
import requests
from bs4 import BeautifulSoup
# Google Api (for Youtube Data Api)
import argparse
from googleapiclient.discovery import build
# other packages
import re
import math
import warnings; warnings.filterwarnings('ignore')

# -------------------- crunchbase scraper class -------------------- #
class youtube_scraper(ScraperBase.scraper):
    """
    Scraper for Youtube. (Version: 1-March-2019)
    """
    def __init__(self, investor, developerKey=None):
        """
        Initialize the Scraper.
        Input:
            investor:              Name of the investor
            developerKey:          Google api key from "https://console.developers.google.com", which has quota
        """
        super(youtube_scraper, self).__init__()
        self.name = 'Youtube'
        self.host = 'www.youtube.com'
        self.investor = investor
        self.videos_id= []
        self.caption_data = []
        self._DEVELOPER_KEY = developerKey if developerKey!=None \
                                           else 'AIzaSyBgRRpuKkvGuGk5QFAGlvgxXoK8AaAYfmE'
        self._WATCH_URL    = 'https://www.youtube.com/watch?v={video_id}'
        self._API_BASE_URL = 'https://www.youtube.com/api/{api_url}'
        
        self.initialize_requests()
        
    def set_header(self):
        """
        Set the header of request.
        """
        self._header['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        self._header['Host']   = self.host
        self._header['Upgrade-Insecure-Requests'] = '1'
        self._session.headers = self._header
        
    def _parse_caption_data(self, r, video_id):
        """
        Parse caption data of 'https://www.youtube.com/watch?v=vidio'.
        Input:
            r:                       Response object of request
            video_id:                Video id of youtube vedio
        """
        agg_data_list = []
        # Use 'BeautifulSoup' package to parse html
        soup = BeautifulSoup(r.text, 'html.parser')
        # Reset parse flag
        self._parseFlag = True
        try:
            caption_raw_list = soup.find_all('text')
            for caption_raw in caption_raw_list:
                # Parse
                start    = caption_raw.attrs['start']
                duration = caption_raw.attrs['dur']
                text     = caption_raw.text
                # Remove <font> tags
                HTML_TAG_REGEX = re.compile(r'<[^>]*>', re.IGNORECASE)
                text           = re.sub(HTML_TAG_REGEX, '', text)
                # Correct single quote to
                text = re.sub('&#39;', '\'', text)
                text = re.sub('&amp;', '&' , text)
                text = re.sub('&gt;' , '>' , text)
                text = re.sub('&lt;' , '<' , text)
                text = re.sub('\n'   , ' ' , text)
                agg_data = {'start'   :start,
                            'duration':duration,
                            'text'    :text}
                agg_data_list.append(agg_data)
        except:
            self._parseFlag = False
            print("Parsing caption data of vedio {} fails.".format(video_id))
            return None
        return agg_data_list
    
    def _interview_text_data(self, caption):
        """
        Extract text info of caption and merge them into a paragraph.
        Input:
            caption:                Dictionary containing caption information
        """
        text_data = []
        for item_list in caption:
            text_data.append(item_list['text'])
        return ' '.join(text_data)
    
    def _get_video_list(self, q, num_result, page_token='', MAX_RESULTS=50):
        """
        Get vedio list of the research result.
        Adapted from 'https://github.com/youtube/api-samples/blob/master/python/search.py'.
        
        Input:
            q:                       Searched item
            num_result:              Total number of searched results obtained
            page_token:              A kind of page id
            MAX_RESULTS:             Maximum number of results of each search
        """
        YOUTUBE_API_SERVICE_NAME = 'youtube'
        YOUTUBE_API_VERSION      = 'v3'
        DEVELOPER_KEY = self._DEVELOPER_KEY
        youtube = build(YOUTUBE_API_SERVICE_NAME, 
                        YOUTUBE_API_VERSION, 
                        developerKey=DEVELOPER_KEY) 
        
        # One time can only maximum 50 results
        if num_result > MAX_RESULTS:
            search_num = MAX_RESULTS                          
        else:
            search_num = num_result
        # Retrieve results matching the specified query term
        search_response = youtube.search().list(q=q,
                                                maxResults=search_num,
                                                part ='id,snippet',
                                                order='relevance',
                                                videoDuration='long',
                                                type='video',
                                                videoCaption='closedCaption',
                                                pageToken=page_token,
                                                relevanceLanguage='en').execute()
        # Add results to the appropriate list
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                title = search_result['snippet']['title']
                # Replace special letter
                title = re.sub('&#39;', '\'', title)
                title = re.sub('&amp;', '&' , title)
                title = re.sub('&gt;' , '>' , title)
                title = re.sub('&lt;' , '<' , title)
                title = re.sub('\n'   , ' ' , title)
                videoId = search_result['id']['videoId']
                publishedAt = search_result['snippet']['publishedAt']
                self.videos_id.append(videoId)
                self.caption_data.append({'title'      :title, 
                                          'videoId'    :videoId, 
                                          'publishedAt':publishedAt})
        # Iteratively get search results
        resultsPerPage = search_response.get('pageInfo')['resultsPerPage']
        num_result = num_result - resultsPerPage
        page_token = search_response.get('nextPageToken')
        # Start to search next page
        if num_result > 0:
            self.videos_id = self._get_video_list(q, num_result, page_token=page_token)
        else:
            return self.videos_id
        return self.videos_id
             
    def scrape_interview_data(self, num_result):
        """
        Scrape interview data of a specific investor.
        
        Input:
            num_result:                     Total number of searched results obtained
        """
        videos_id = self._get_video_list(self.investor, num_result)
        basis     = math.ceil(num_result / 100)
        
        # Get the caption of each vedio iteratively using "get api"
        for i, video_id in enumerate(videos_id):
            fetched_site = self._session.get(self._WATCH_URL.format(video_id=video_id)).text
            # Find api token from html
            timedtext_url_start = fetched_site.find('timedtext') 
            timedtext_url_end   = timedtext_url_start + fetched_site[timedtext_url_start:].find('"')
            api_url = fetched_site[timedtext_url_start:timedtext_url_end] \
                                  .replace('\\u0026', '&')\
                                  .replace('\\', '')
            api_url = self._API_BASE_URL.format(api_url=api_url)
            
            # Get the raw caption text
            r = self._session.get(api_url)
            # Parse caption data
            caption = self._parse_caption_data(r, video_id)
            # Drop the data, if it cannot be parsed
            if not self._parseFlag:
                # It will be deleted in InvestorDataFrame module
                caption_text = ''
            else:
                caption_text = self._interview_text_data(caption)
            self.caption_data[i]['caption']      = caption
            self.caption_data[i]['caption_text'] = caption_text
            
            if i % basis == 0:
                percentage = 100 * i / num_result
                print(f'{percentage:.2f}% scraping task has been finished.', end = '\r')
        print('The scraping task has been finished!\t\t\t')
        self.caption_data = {self.investor: self.caption_data}