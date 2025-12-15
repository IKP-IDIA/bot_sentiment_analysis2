import requests
from bs4 import BeautifulSoup
# from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
import xml.etree.ElementTree as ET
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import json
import uuid

THAI_SENTIMENT_LEXICON = {
    # คำเชิงบวกมาก (0.8 - 1.0)
    "ดีเยี่ยม": 1.0, "เยี่ยมยอด": 1.0, "สุดยอด": 1.0, "ยอดเยี่ยม": 1.0, 
    "เจริญ": 0.9, "รุ่งเรือง": 0.9, "เติบโต": 0.9, "พุ่ง": 0.9, "ทะยาน": 0.9,
    "สำเร็จ": 0.8, "ชนะ": 0.8, "ได้": 0.8, "ดี": 0.8, "เยี่ยม": 0.9,
    
    # คำเชิงบวกปานกลาง (0.4 - 0.7)
    "ชอบ": 0.7, "พอใจ": 0.7, "ยินดี": 0.7, "ดีใจ": 0.7, "สดใส": 0.7,
    "ขึ้น": 0.6, "เพิ่ม": 0.6, "ดีขึ้น": 0.6, "ฟื้นตัว": 0.6, "แข็งแกร่ง": 0.6,
    "มั่นคง": 0.5, "ราบรื่น": 0.5, "ปกติ": 0.4, "โอเค": 0.4,
    
    # คำเชิงลบมาก (-0.8 ถึง -1.0)
    "แย่มาก": -1.0, "ล้มเหลว": -1.0, "เจ๊ง": -1.0, "ล่มสลาย": -1.0, "วิกฤต": -1.0,
    "ทุจริต": -0.9, "โกง": -0.9, "ฉ้อโกง": -0.9, "คอร์รัปชั่น": -0.9, "หลอกลวง": -0.9,
    "ขาดทุน": -0.9, "ตกต่ำ": -0.9, "ย่ำแย่": -0.9, "ตกกระป๋อง": -0.9, "ดิ่ง": -0.9,
    "แย่": -0.8, "สแกม": -0.8, "สแกมเมอร์": -0.8, "เสีย": -0.8, "เลวร้าย": -0.8,
    "ฟอกเงิน": -0.9, "ฟอกเงิน": -0.9, "ผิดกฏหมาย": -0.9,
    
    # คำเชิงลบปานกลาง (-0.4 ถึง -0.7)
    "ปัญหา": -0.7, "กังวล": -0.7, "ห่วง": -0.7, "เสี่ยง": -0.7, "อันตราย": -0.7,
    "ลดลง": -0.6, "ลด": -0.6, "หด": -0.6, "ตก": -0.6, "ลง": -0.6,
    "อ่อนแอ": -0.5, "ชะลอ": -0.5, "ซบเซา": -0.5, "ซึม": -0.5, "ติดขัด": -0.5,
    "แพง": -0.4, "เหนื่อย": -0.4, "ยาก": -0.4,
    
    # คำเกี่ยวกับเศรษฐกิจและการเงิน
    "กำไร": 0.8, "รายได้": 0.6, "เงินทุน": 0.5, "ลงทุน": 0.5,
    "หนี้": -0.6, "ขาดดุล": -0.7, "เงินเฟ้อ": -0.6, "ว่างงาน": -0.7,
    
    # คำเกี่ยวกับหุ้น
    "แกว่ง": 0.0, "คาดการณ์": 0.0, "ประเมิน": 0.0, "วิเคราะห์": 0.0,
    "ขาย": -0.3, "ถือ": 0.1, "ซื้อ": 0.4, "แนะนำซื้อ": 0.7,
    
    # คำเสริมความหมาย (Intensifiers)
    "มาก": 1.2, "มากมาย": 1.2, "สุด": 1.3, "ที่สุด": 1.3, "เกินไป": 1.2,
    "ไม่": -1.5, "ไม่ใช่": -1.5, "ไม่ได้": -1.5,
}
THAI_STOPWORDS = set(thai_stopwords())

def get_google_news(keyword, lang="th", limit=20):
    """
    Fetch the lastest new for a given stock from google.com
    """
    if lang == "th":
        url = f"https://news.google.com/rss/search?q={keyword}&hl=th&gl=TH&ceid=TH:th"
    elif lang =="en":
        url =  f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    else:
        print("Unsuported language.")
        return []

    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response  = requests.get(url, headers=headers)
        response.raise_for_status() # check HTTP status
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    
    # Use XML Parsiing for analyst RSS Feed 
    soup = ET.fromstring(response.content)
    news_list = []

    #RSS News items อยู่ใน <item> in <channel>
    for item in soup.findall('./channel/item')[:limit]:
        title = item.find('title').text if item.find('title') is not None else 'N/A'
        link = item.find('link').text if item.find('link') is not None else 'N/A'
        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'N/A'

        news_list.append({
            'title':title,
            'link': link,
            'pubDate': pub_date
        })
    return news_list

def parse_news(news_list):
    """
    Parse the news table and extract relevant information
    """
    parsed_news = []
    for news_item in news_list:
        title = news_item['title']
        pub_date_str = news_item['pubDate']

        #Attemp to parse the date and time from the RRS pubDate string
        try:
            # Format 'Wed, 20 Nov 2025 08:31:46 GMT'
            dt = datetime.strptime(pub_date_str,'%a, %d %b %Y %H:%M:%S %Z')
            current_date = dt.strftime("%d-%b-%y")
            time = dt.strftime("%H:%M:%S")
        
        except ValueError:
            current_date = date.today().strftime("%d-%b-%y")
            time = "N/A"

        # Append the formatted data
        parsed_news.append([current_date, time,title])

    return parsed_news

def analyze_sentiment(parsed_news):
    """
    Perform sentiment analysis on the parsed news
    """
    for news in parsed_news:
        title = news[2]
        
        # 1. Word Segmentation (Tokenization)
        # Use the default dictionary for segmentation
        tokens = word_tokenize(title, engine='newmm') 
        
        # 2. Cleanup and Score Calculation
        total_score = 0
        word_count = 0
        
        # Improvment : Implement simple intensification logic
        intensifier_multiplier = 1.0

        for token in tokens:
            # Clean up the token (remove punctuation, normalize)
            token = token.strip()

            # Skip stop words and short tokens
            if not token or token in THAI_STOPWORDS or len(token) < 2:
                # Reset multiplier if hit a stopword/filler 
                intensifier_multiplier =1.0
                continue

            # 3. Lexicon Lookup and scoring
            score = THAI_SENTIMENT_LEXICON.get(token, 0)

            # Apply multiplier to the word's score
            final_score = score * intensifier_multiplier

            if score != 0:
                total_score += final_score
                word_count += 1
        
        # Reset multiplier after use (or if it was a negative modifier, the effect is already applied)
        intensifier_multiplier = 1.0

        # Calculate average polarity
        polarity = total_score / word_count if word_count > 0 else 0
        
        # จำกัดช่วงคะแนน
        polarity = max(-1.0, min(1.0, polarity))
    
        # กำหนด label
        if polarity >= 0:
            label = 'ไม่พบข้อมูลเชิงลบ'
        #elif polarity < -0.1:
        #    label = 'negative'
        else:
            label = 'พบข้อมูลเชิงลบ'
        
        # 4. Append results to the current news item (list)
        news.append(polarity)
        news.append(label)
    
    return parsed_news

def plot_sentiment(df, ticker, avg_sentiment):
    """
    Plot the sentiment analysis results on separate figures
    """
    # Scatter plot of sentiment vs news number
    plt.figure(figsize=(12, 6))
    df['news_num'] = range(1, len(df) + 1)  # Add a news number column
    plt.scatter(df['news_num'], df['sentiment'], alpha=0.6, label='News Sentiment')
    plt.plot(df['news_num'], df['sentiment'], alpha=0.4)  # Adding a line for trend visibility
    plt.axhline(y=avg_sentiment, color='g', linestyle='--', alpha=0.8, label=f'Average Sentiment: {avg_sentiment:.2f}')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Neutral Sentiment')
    plt.title(f'Sentiment Analysis of {ticker} News')
    plt.xlabel('News Number (Chronological Order)')
    plt.ylabel('Sentiment Polarity')
    for i, (news_num, sentiment, title) in enumerate(zip(df['news_num'], df['sentiment'], df['title'])):
        if i % 5 == 0:  # Annotate every 5th point to avoid clutter
            plt.annotate(f"{news_num}", (news_num, sentiment), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.legend()
    plt.show()  # Show the first plot on a separate page

    # Histogram of sentiment distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['sentiment'], bins=20, edgecolor='black')
    plt.axvline(x=avg_sentiment, color='r', linestyle='--', label=f'Average: {avg_sentiment:.2f}')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()  # Show the second plot on a separate page

    # Pie chart of positive vs negative sentiments

    positive = (df['sentiment'] > 0).sum()
    negative = (df['sentiment'] < 0).sum()
    neutral = (df['sentiment'] == 0).sum()

    pie_colors = ['#4CAF50', '#F44336', '#BDBDBD']
    plt.figure(figsize=(12,6))
    plt.pie([positive, negative, neutral], labels=['Positive', 'Negative', 'Neutral'],colors=pie_colors ,autopct='%1.1f%%')
    plt.title('Proportion of Positive vs Negative Sentiments')
    plt.show()  # Show the third plot on a separate page

def send_results_to_api(json_data, api_url):
    """
    Sends the generated JSON data (Micro-Payload) to a specified API endpoint.
    """
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, json=json_data, timeout=10)
        response.raise_for_status() 
        
        return {
            "status": "success", 
            "message": f"Data sent successfully. Status code: {response.status_code}",
            "response_data": response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error", 
            "message": "API request failed. Ensure the FastAPI server is running.",
            "details": str(e)
        }

def main(ticker):
    """
    Main function to run the sentiment analysis
    """
    news_table = get_google_news(ticker, lang="th")

    if not news_table:
        print(f"No news found for '{ticker}'. Skipping Analysis.")
        return None 

    parsed_news = parse_news(news_table)
    analyzed_news = analyze_sentiment(parsed_news)

    df = pd.DataFrame(analyzed_news, columns=['date', 'time', 'title', 'sentiment','label'])

    # Calculate average sentiment
    cnt = (df['sentiment'] < 0.0).sum()
    avg_sentiment = df['sentiment'].mean()

    sentiment_result = (
        "ไม่พบข้อมูลเชิงลบ" if cnt == 0 else 
        #"ไม่พบข้อมูลเชิงลบ" if avg_sentiment >= -0.1 else
        "พบข้อมูลเชิงลบจำนวน " + str(cnt) +" รายการ"
    )

    API_ENDPOINT = "http://127.0.0.1:8001/api/sentiment" 

    json_payload = {
        "analysis_id": str(uuid.uuid4()), # ใช้ uuid ที่ import มา
        "analysis_date": datetime.now().isoformat(),
        "keyword": ticker,
        "total_articles": len(df),
        "average_sentiment": float(f"{avg_sentiment:.4f}"),
        "overall_label": sentiment_result,
        # (ไม่มี news_articles)
    }

    api_response = send_results_to_api(json_payload, API_ENDPOINT)

    # Display Results
    print(f"\n{'='*70}")
    print(f"Analyzing: {ticker}")
    print(f"{'='*70}")
    
    print(df.to_string())
    print(f"\n Average sentiment: {avg_sentiment:.2f}")
    print(sentiment_result)

    # ADD: API Response Print
    print("\n--- API Submission Status ---")
    print(json.dumps(api_response, indent=4))

    # Plot sentiments including average
    # plot_sentiment(df, ticker, avg_sentiment)

    return df

#if __name__ == "__main__":
def call_function(ticker):
    #ticker = input("Enter keywords: ")
    #main(ticker)
    search_keyword = ticker.strip()
    final_df = main(search_keyword)
    
    cnt = (final_df['sentiment'] < 0).sum()
    print(cnt)
    avg_sentiment = final_df['sentiment'].mean()
    sentiment_result = (
        "ไม่พบข้อมูลเชิงลบ" if cnt == 0 else 
        #"ไม่พบข้อมูลเขิงลบ" if avg_sentiment >= -0.1 else
        "พบข้อมูลเชิงลบจำนวน " + str(cnt) + " รายการ"
    )
    
    if final_df is not None:
        try:
            file_name = f'{search_keyword.replace(" ", "_")}_thai_sentiment.csv'
            final_df.to_csv(file_name, index=False)
            print(f"\n Saved detailed results to {file_name}")
        except Exception as e :
            print(f"\n Error saving file: {e}")
    
    return sentiment_result

#df.to_csv(f'{ticker}_sentiment.csv', index=False)
#print(f"Saved to {ticker}_sentiment.csv")
# if __name__=="__main__": 
#     keywords = input("Enter keywords : ")
#     articles = get_google_news(keywords)

#     if articles:
#         print(f"\n Found {len(articles)} articles for '{keywords}'")
#         for i, article in enumerate(articles[:5]): #show first 5 news 
#             print(f"--- Article {i+1} ---")
#             print(f"Title: {article['title']}")
#             print(f"Date: {article['pubDate']}")
#             print(f"Link: {article['link']}")
#     else:
#         print("No articles found.")
