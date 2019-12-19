# coding: utf-8
# -------------------- Package import -------------------- #
import os
import requests
import multiprocessing

# -------------------- scraper class -------------------- #
class scraper:
    """
    Basic scraper class.
    """
    def __init__(self):
        self.host       = 'NULL'
        self._session   = requests.session()
        self._header    = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'}
        self._parseFlag = True
        self._linkFlag  = True
        
    def set_header(self):
        """
        Set the header of request.
        """
        raise NotImplementedError
        
    def initialize_requests(self):
        """
        Initialize headers and cookies for requests.
        """
        self.set_header()
        # Preserve the cookies in the session
        res = self._session.get(url='https://'+self.host, verify=False)
        if res.status_code != 200:
            # Change proxy until link to the page
            print(res.status_code, ': Fill to access, please change a proxy!')
            self._change_proxy_requests()
        else:
            print('Successful to access!')
            return 
        
    def _change_proxy_requests(self, PROXY=None):
        """
        Change proxy of request session when being blocked. 
        """
        # Inline function to input proxy
        def inputProxy(proxyPoolFile):
            while True:
                PROXY = input('Proxy in "IP:PORT", e.g. "185.197.30.48:3128". Find more on "https://www.us-proxy.org".\n')
                proxies = {'http' : 'http://' + PROXY,
                           'https': 'https://'+ PROXY}
                self._session.proxies = proxies
                try:
                    res = self._session.get(url='https://'+self.host, verify=False)
                except:
                    continue
                if res.status_code == 200:
                    # Write useful proxy into file
                    with open(proxyPoolFile, 'a', encoding='utf-8') as file:
                        file.write(PROXY + '\n')
                    print('Successful to access! Proxy: {}'.format(PROXY))
                    break
            return PROXY

        proxyPoolFile = './setting/proxyPool'
        # lines = []
        # If the file of 'proxyPool' exists
        if os.path.exists(proxyPoolFile):
            with open(proxyPoolFile, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines = [i.strip() for i in lines]
            # If the file is not empty
            if len(lines) != 0:  
                for i, PROXY in enumerate(lines):
                    proxies = {'http' : 'http://' + PROXY,
                               'https': 'https://'+ PROXY}
                    self._session.proxies = proxies
                    try:
                        res = self._session.get(url='https://'+self.host, verify=False)
                    except:
                        lines[i] = None
                        continue
                    if res.status_code == 200:
                        # Return useful proxy
                        print('Successful to access! Proxy: {}'.format(PROXY))
                        FinalProxy = PROXY
                        break
                    else:
                        # Delete useless/expired proxy
                        print('Can not use: {}'.format(PROXY))
                        lines[i] = None
                # Remain useful lines
                usefulProxy = [proxyInd for proxyInd in lines if type(proxyInd) != type(None)]
                with open(proxyPoolFile, 'w', encoding='utf-8') as file:
                    for line in usefulProxy:
                        file.write(line + '\n')
                if len(usefulProxy) == 0:
                    FinalProxy = inputProxy(proxyPoolFile)
            else:
                FinalProxy = inputProxy(proxyPoolFile)
        else:
            FinalProxy = inputProxy(proxyPoolFile)
        return FinalProxy
    
    def _multi_process(self, func, urlList):
        """
        Multi-process scraping tasks at the same time.
        
        Input:
            func:                       Functions used for scraping
            urlList:                    List of scraped urls
        """
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        pool.map_async(func, urlList)
        pool.close()
        pool.join()

    def save(self, path, data):
        """
        Pickle the data to the local.
        Input:
            path:                       The path of to save the file
            data:                       Data to be saved
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def parse(self, respond):
        """
        Parse the data from the request response.
        """
        pass