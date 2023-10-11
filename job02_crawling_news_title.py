# from selenium import webdriver
#
# from os import path
#
# from selenium.common.exceptions import NoSuchDriverException
# from selenium.webdriver.common.options import BaseOptions
# from selenium.webdriver.common.selenium_manager import SeleniumManager
# from selenium.webdriver.common.service import Service
#
# from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
# import pandas as pd
# from selenium.webdriver.chromium.webdriver import ChromiumDriver
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
#
# from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium.webdriver.chrome.options import Options as ChromeOptions
# from webdriver_manager.chrome import ChromeDriverManager
#
#
#
# import re
#
# import time
#
# import datetime
#
# url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1'
#
# options=ChromeOptions()
#
# user_agent=
#
# options.add_argument('lang=ko_KR')
#
# #path = SeleniumManager().driver_location(options) if path is None else path
#
# #driver=webdriver.Chrome(service='./chromewebdriver',options='option')
#
# driver.get(url)
import re
import time
import datetime

import pandas
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

options = ChromeOptions()
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
options.add_argument('user-agent=' + user_agent)
options.add_argument("lang=ko_KR")
# options.add_argument('headless')
# options.add_argument('window-size=1920x1080')
# options.add_argument("disable-gpu")
# options.add_argument("--no-sandbox")

# 크롬 드라이버 최신 버전 설정
service = ChromeService(executable_path=ChromeDriverManager().install())

url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1'

category=['Politics','Economic','Social','Culture','World','IT']

# chrome driver
driver = webdriver.Chrome(service=service, options=options)  # <- options로 변경
driver.get(url)

#xpath='//*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a'

#title=driver.find_element('xpath','//*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a').text

titles=[]

pages=[146,110,110,75,110,72]

# for i in range(1, 5):  # 1부터 5까지
#     for j in range(1, 6):  # 1부터 5까지
#         #xpath = '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[{}]/a'.format(i, j, i)
#         title = driver.find_element('xpath','//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(i, j)).text
#         title=re.compile('[^가-힣]').sub(' ',title)
#         titles.append(title)

        #title = driver.find_element(xpath).text

df_titles=pd.DataFrame()

for l in range(6):

    section_url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(l)
    #driver.get(url)
    titles=[]

    for k in range(1,3):

        url = section_url+'#&date=%2000:00:00&page={}'.format(k)
        driver.get(url)
        time.sleep(0.5)
        for i in range(1, 5):  # 1부터 5까지
            for j in range(1, 6):  # 1부터 5까지
                # xpath = '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[{}]/a'.format(i, j, i)
                title = driver.find_element('xpath', '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(i, j)).text
                title = re.compile('[^가-힣]').sub(' ', title)
                titles.append(title)

    df_section_title=pandas.DataFrame(titles,columns=['titles'])
    df_section_title['category']=category[l]

    df_titles=pd.concat([df_titles,df_section_title],ignore_index=True)

df_titles.to_csv('./crawling_data/crawling_data.csv',index=False)


print(df_titles.head())
df_titles.info()

cnt=df_titles['category'].value_counts()

print(cnt)
#print(title)

# //*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[4]/dl/dt[2]/a

# //*[@id="section_body"]/ul[2]/li[1]/dl/dt[2]/a

# //*[@id="section_body"]/ul[4]/li[5]/dl/dt[2]/a

# //*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a