import streamlit as st
import yfinance as yf
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gazpacho
import matplotlib as plt
st.title("Welcome to pAIsa, the Indian AI Investment Bank!")

# Create a dropdown menu
service_type = st.selectbox("Select a service type:", [
    "Asset Management Services",
    "Mergers and Acquisition Advisory Services",
    "Research Management Services",
    "Risk Analysis Services",
    "Structured Financial Management Services",
    "Exit Strategies",
    "IPO Management Services",
    "Private Equity Services",
    "Venture Capital Services",
    "Debt Capital Markets Services",
    "Equity Capital Markets Services",
    "Restructuring Services",
    "Corporate Finance Services",
    "Financial Planning Services",
    "Stock Data Pulling Services"
])
if service_type == "Mergers and Acquisition Advisory Services":
    # Dictionary to map exchanges to suffixes
    exchange_suffixes = {
        "NYSE": "",
        "NASDAQ": "",
        "BSE": "",
        "NSE": ".NS",
        "Cboe Indices": ".CI",
        "Chicago Board of Trade (CBOT)***": ".CBT",
        "Chicago Mercantile Exchange (CME)***": ".CME",
        "Dow Jones Indexes": ".DJ",
        "Nasdaq Stock Exchange": ".NSQ",
        "ICE Futures US": ".NYB",
        "New York Commodities Exchange (COMEX)***": ".CMX",
        "New York Mercantile Exchange (NYMEX)***": ".NYM",
        "Options Price Reporting Authority (OPRA)": ".OPR",
        "OTC Markets Group**": ".OTC",
        "S & P Indices": ".SP",
        "Buenos Aires Stock Exchange (BYMA)": ".BA",
        "Vienna Stock Exchange": ".VI",
        "Australian Stock Exchange (ASX)": ".AX",
        "Cboe Australia": ".CBA",
        "Euronext Brussels": ".BR",
        "Sao Paolo Stock Exchange (BOVESPA)": ".SA",
        "Canadian Securities Exchange": ".CN",
        "Cboe Canada": ".CBOE",
        "Toronto Stock Exchange (TSX)": ".TO",
        "TSX Venture Exchange (TSXV)": ".TV",
        "Santiago Stock Exchange": ".SN",
        "Shanghai Stock Exchange": ".SS",
        "Shenzhen Stock Exchange": ".SZ",
        "Prague Stock Exchange Index": ".PR",
        "Nasdaq OMX Copenhagen": ".CO",
        "Egyptian Exchange Index (EGID)": ".CA",
        "Nasdaq OMX Tallinn": ".TL",
        "Cboe Europe": ".CE",
        "Euronext": ".EU",
        "Nasdaq OMX Helsinki": ".HE",
        "Euronext Paris": ".PA",
        "Berlin Stock Exchange": ".BE",
        "Bremen Stock Exchange": ".BM",
        "Dusseldorf Stock Exchange": ".DU",
        "Frankfurt Stock Exchange": ".F",
        "Hamburg Stock Exchange": ".HM",
        "Hanover Stock Exchange": ".HA",
        "Munich Stock Exchange": ".MU",
        "Stuttgart Stock Exchange": ".SG",
        "Deutsche Boerse XETRA": ".DE",
        "Collectable Indices": ".REGA",
        "Cryptocurrencies": "",
        "Currency Rates": ".X",
        "MSCI Indices": ".MSCI",
        "Athens Stock Exchange (ATHEX)": ".AT",
        "Hang Seng Indices": ".HSI",
        "Hong Kong Stock Exchange (HKEX)*": ".HK",
        "Budapest Stock Exchange": ".BD",
        "Nasdaq OMX Iceland": ".IC",
        "Bombay Stock Exchange": ".BO",
        "National Stock Exchange of India": ".NS",
        "Indonesia Stock Exchange (IDX)": ".JK",
        "Euronext Dublin": ".ID",
        "Tel Aviv Stock Exchange": ".TA",
        "EuroTLX": ".TLX",
        "Italian Stock Exchange": ".MI",
        "Nikkei Indices": ".NIKKEI",
        "Tokyo Stock Exchange": ".T",
        "Boursa Kuwait": ".KW",
        "Nasdaq OMX Riga": ".RG",
        "Nasdaq OMX Vilnius": ".VL",
        "Malaysian Stock Exchange": ".KL",
        "Mexico Stock Exchange (BMV)": ".MX",
        "Euronext Amsterdam": ".AS",
        "New Zealand Stock Exchange (NZX)": ".NZ",
        "Oslo Stock Exchange": ".OL",
        "Philippine Stock Exchange Indices": ".PS",
        "Warsaw Stock Exchange": ".WA",
        "Euronext Lisbon": ".LS",
        "Qatar Stock Exchange": ".QA",
        "Bucharest Stock Exchange": ".RO",
        "Singapore Stock Exchange (SGX)": ".SI",
        "Johannesburg Stock Exchange": ".JO",
        "Korea Stock Exchange": ".KS",
        "KOSDAQ": ".KQ",
        "Madrid SE C.A.T.S.": ".MC",
        "Saudi Stock Exchange (Tadawul)": ".SAU",
        "Nasdaq OMX Stockholm": ".ST",
        "Swiss Exchange (SIX)": ".SW",
        "Taiwan OTC Exchange": ".TWO",
        "Taiwan Stock Exchange (TWSE)": ".TW",
        "Stock Exchange of Thailand (SET)": ".BK",
        "Borsa İstanbul": ".IS",
        "Dubai Financial Market": ".AE",
        "Cboe UK": ".CUK",
        "FTSE Indices": ".FTSE",
        "London Stock Exchange": ".L",
        "Caracas Stock Exchange": ".CR"
    }
    
    # Dictionary mapping exchanges to their major indices or benchmarks
    exchange_indices = {
        "NYSE": "^GSPC",  # S&P 500
        "NASDAQ": "^IXIC",  # NASDAQ Composite
        "BSE": "^BSESN",  # SENSEX
        "NSE": "^NSEI",  # NIFTY 50
        "Cboe Indices": "^VIX",  # Cboe Volatility Index (VIX)
        "Chicago Board of Trade (CBOT)***": "^VIX",  # S&P 500 VIX (example, needs actual ticker)
        "Chicago Mercantile Exchange (CME)***": "CME",  # CME Group (example, needs actual ticker)
        "Dow Jones Indexes": "^DJI",  # Dow Jones Industrial Average
        "Nasdaq Stock Exchange": "^IXIC",  # NASDAQ Composite
        "ICE Futures US": "^RUT",  # Russell 2000
        "New York Commodities Exchange (COMEX)***": "GC=F",  # COMEX Gold (example, needs actual ticker)
        "New York Mercantile Exchange (NYMEX)***": "CL=F",  # NYMEX Crude Oil (example, needs actual ticker)
        "Options Price Reporting Authority (OPRA)": "OPRA",  # OPRA Index (example, needs actual ticker)
        "OTC Markets Group**": "OTCM",  # OTCQX Best Market (example, needs actual ticker)
        "S & P Indices": "^GSPC",  # S&P 500
        "Buenos Aires Stock Exchange (BYMA)": "^MERV",  # MERVAL
        "Vienna Stock Exchange": "^ATX",  # ATX
        "Australian Stock Exchange (ASX)": "^AXJO",  # S&P/ASX 200
        "Cboe Australia": "^XVI",  # S&P/ASX 200 VIX (example, needs actual ticker)
        "Euronext Brussels": "^BFX",  # BEL 20
        "Sao Paolo Stock Exchange (BOVESPA)": "^BVSP",  # IBOVESPA
        "Canadian Securities Exchange": "^GSPTSE",  # S&P/TSX Composite
        "Cboe Canada": "^VIXC",  # S&P/TSX Composite VIX (example, needs actual ticker)
        "Toronto Stock Exchange (TSX)": "^GSPTSE",  # S&P/TSX Composite
        "TSX Venture Exchange (TSXV)": "^JX",  # TSX Venture Composite (example, needs actual ticker)
        "Santiago Stock Exchange": "^IPSA",  # IPSA
        "Shanghai Stock Exchange": "000001.SS",  # SSE Composite Index
        "Shenzhen Stock Exchange": "399001.SZ",  # SZSE Component Index
        "Prague Stock Exchange Index": "^PX",  # PX
        "Nasdaq OMX Copenhagen": "^OMXC25",  # OMX Copenhagen 25
        "Egyptian Exchange Index (EGID)": "^EGX30.CA",  # EGX 30
        "Nasdaq OMX Tallinn": "^OMXTGI",  # OMX Tallinn
        "Cboe Europe": "^STOXX50E",  # EURO STOXX 50
        "Euronext": "^N100",  # Euronext 100
        "Nasdaq OMX Helsinki": "^OMXH25",  # OMX Helsinki 25
        "Euronext Paris": "^FCHI",  # CAC 40
        "Berlin Stock Exchange": "^GDAXI",  # DAX
        "Bremen Stock Exchange": "BREXIT",  # BREXIT (example, needs actual ticker)
        "Dusseldorf Stock Exchange": "^GDAXI",  # DAX
        "Frankfurt Stock Exchange": "^GDAXI",  # DAX
        "Hamburg Stock Exchange": "BREXIT",  # BREXIT (example, needs actual ticker)
        "Hanover Stock Exchange": "BREXIT",  # BREXIT (example, needs actual ticker)
        "Munich Stock Exchange": "^GDAXI",  # DAX
        "Stuttgart Stock Exchange": "BREXIT",  # BREXIT (example, needs actual ticker)
        "Deutsche Boerse XETRA": "^GDAXI",  # DAX
        "Collectable Indices": "COLLECT",  # COLLECT (example, needs actual ticker)
        "Cryptocurrencies": "CRYPTO",  # CRYPTO (example, needs actual ticker)
        "Currency Rates": "CURRENCY",  # CURRENCY (example, needs actual ticker)
        "MSCI Indices": "MSCI",  # MSCI (example, needs actual ticker)
        "Athens Stock Exchange (ATHEX)": "^ATG",  # ASE
        "Hang Seng Indices": "^HSI",  # HANG SENG
        "Hong Kong Stock Exchange (HKEX)*": "^HSI",  # HANG SENG
        "Budapest Stock Exchange": "^BUX",  # BUX
        "Nasdaq OMX Iceland": "^OMXICELAND",  # OMX
        "Bombay Stock Exchange": "^BSESN",  # BSE
        "National Stock Exchange of India": "^NSEI",  # NSE
        "Indonesia Stock Exchange (IDX)": "^JKSE",  # IDX
        "Euronext Dublin": "^ISEQ",  # INDEX (example, needs actual ticker)
        "Tel Aviv Stock Exchange": "^TA125.TA",  # TEL AVIV
        "EuroTLX": "^TLX",  # EURO TLX (example, needs actual ticker)
        "Italian Stock Exchange": "FTSEMIB.MI",  # ITALY (example, needs actual ticker)
        "Nikkei Indices": "^N225",  # NIKKEI
        "Tokyo Stock Exchange": "^TPX",  # TOKYO (example, needs actual ticker)
        "Boursa Kuwait": "^KWSE",  # KW (example, needs actual ticker)
        "Nasdaq OMX Riga": "^OMXRGI",  # RIGA
        "Nasdaq OMX Vilnius": "^OMXVGI",  # VILNIUS
        "Malaysian Stock Exchange": "^KLSE",  # MALAYSIA (example, needs actual ticker)
        "Mexico Stock Exchange (BMV)": "^MXX",  # BMV (example, needs actual ticker)
        "Euronext Amsterdam": "^AEX",  # EUROPE (example, needs actual ticker)
        "New Zealand Stock Exchange (NZX)": "^NZ50",  # NZX (example, needs actual ticker)
        "Oslo Stock Exchange": "^OSEAX",  # OSLO (example, needs actual ticker)
        "Philippine Stock Exchange Indices": "^PSEi",  # PHILIPPINES (example, needs actual ticker)
        "Warsaw Stock Exchange": "^WIG",  # WSE (example, needs actual ticker)
        "Euronext Lisbon": "^PSI20",  # LISBON (example, needs actual ticker)
        "Qatar Stock Exchange": "^QSI",  # QATAR (example, needs actual ticker)
        "Bucharest Stock Exchange": "^BET",  # BUCHAREST (example, needs actual ticker)
        "Singapore Stock Exchange (SGX)": "^STI",  # SGX (example, needs actual ticker)
        "Johannesburg Stock Exchange": "^J203.JO",  # JOHANNESBURG (example, needs actual ticker)
        "Korea Stock Exchange": "^KS11",  # KOREA (example, needs actual ticker)
        "KOSDAQ": "^KQ11",  # KOSDAQ (example, needs actual ticker)
        "Madrid SE C.A.T.S.": "^IBEX",  # MADRID (example, needs actual ticker)
        "Saudi Stock Exchange (Tadawul)": "^TASI.SR",  # TADAWUL (example, needs actual ticker)
        "Nasdaq OMX Stockholm": "^OMX",  # OMX (example, needs actual ticker)
        "Swiss Exchange (SIX)": "^SSMI",  # SIX (example, needs actual ticker)
        "Taiwan OTC Exchange": "^TWO",  # OTC (example, needs actual ticker)
        "Taiwan Stock Exchange (TWSE)": "^TWII",  # TWSE (example, needs actual ticker)
        "Stock Exchange of Thailand (SET)": "^SET.BK",  # SET (example, needs actual ticker)
        "Borsa İstanbul": "^XU100",  # ISTANBUL (example, needs actual ticker)
        "Dubai Financial Market": "^DFMGI",  # DFM (example, needs actual ticker)
        "Cboe UK": "^UKX",  # UK (example, needs actual ticker)
        "FTSE Indices": "^FTSE",  # FTSE (example, needs actual ticker)
        "London Stock Exchange": "^FTSE",  # LSE (example, needs actual ticker)
        "Caracas Stock Exchange": "^IBC"
    }
    
    market_cap_categories = {
        "Mega-cap": 200e9,
        "Large-cap": 10e9,
        "Mid-cap": 2e9,
        "Small-cap": 500e6,
        "Micro-cap": 50e6,
        "Nano-cap": 0
    }

    # Set up the Streamlit app
st.title("Financial Data Analysis and NLP News Sentiment")

# Get user input for ticker symbols
acquirer_ticker = st.text_input("Enter acquirer's stock ticker:")
acquirer_exchange = st.selectbox("Select acquirer's exchange:", list(exchange_suffixes.keys()))

acquiree_ticker = st.text_input("Enter acquiree's stock ticker:")
acquiree_exchange = st.selectbox("Select acquiree's exchange:", list(exchange_suffixes.keys()))

date_range = st.selectbox("Select date range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])

# Define date ranges
if date_range == "1d":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=1)
elif date_range == "5d":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=5)
elif date_range == "1mo":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=30)
elif date_range == "3mo":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=90)
elif date_range == "6mo":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=180)
elif date_range == "1y":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=365)
elif date_range == "2y":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=730)
elif date_range == "5y":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=1825)
elif date_range == "10y":
    start_date = pd.to_datetime('today') - pd.Timedelta(days=3650)
elif date_range == "max":
    start_date = pd.to_datetime('1924-01-01')
end_date = pd.to_datetime('today')

# Function to fetch data from yfinance
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

# Fetch data if tickers and exchanges are provided
if acquirer_ticker and acquirer_exchange and acquiree_ticker and acquiree_exchange:
    acquirer_ticker_with_suffix = acquirer_ticker + exchange_suffixes[acquirer_exchange]
    acquiree_ticker_with_suffix = acquiree_ticker + exchange_suffixes[acquiree_exchange]
    acquirer_data = fetch_data(acquirer_ticker_with_suffix, start=start_date, end=end_date)
    acquiree_data = fetch_data(acquiree_ticker_with_suffix, start=start_date, end=end_date)
else:
    st.write("Enter all the data!")

# Calculate estimated debt volume and other metrics
def calculate_metrics(data):
    data['Estimated Debt Volume'] = (data['Close'] - data['Adj Close']) * data['Volume']
    data['Average Total Assets'] = data['Adj Close'] * data['Volume']
    data['Asset Turnover Ratio'] = data['Volume'] / data['Average Total Assets']
    data['EBIT'] = (data['Volume'] * data['Close']) - (data['Volume'] * data['Close']) - ((data['Volume'] * data['Close']) * (data['Close'] - data['Open']) / data['Volume'])
    data['Interest Rate'] = 0.08
    data['Corporate Tax'] = 0.235
    
    # Calculate various ratios
    data['Debt-to-Equity Ratio'] = data['Estimated Debt Volume'] / data['Adj Close']
    data['Current Ratio'] = data['Adj Close'] / data['Estimated Debt Volume']
    data['Interest Coverage Ratio'] = data['Adj Close'] / (data['Estimated Debt Volume'] * 0.05)
    data['Debt-to-Capital Ratio'] = data['Estimated Debt Volume'] / (data['Adj Close'] + data['Estimated Debt Volume'])
    data['Price-to-Earnings Ratio'] = data['Close'] / data['Adj Close']
    data['Price-to-Book Ratio'] = data['Close'] / data['Adj Close']
    data['Return on Equity (ROE)'] = (data['Close'] - data['Open']) / data['Adj Close']
    data['Return on Assets (ROA)'] = (data['Close'] - data['Open']) / data['Volume']
    data['Earnings Yield'] = data['Adj Close'] / data['Close']
    data['Dividend Yield'] = data['Adj Close'] / data['Close']
    data['Price-to-Sales Ratio'] = data['Close'] / data['Volume']
    data['Enterprise Value-to-EBITDA Ratio'] = (data['Close'] * data['Volume']) / (data['Adj Close'] * 0.05)
    data['Inventory Turnover Ratio'] = data['Volume'] / (data['Close'] - data['Open'])
    data['Receivables Turnover Ratio'] = data['Volume'] / (data['Close'] - data['Open'])
    data['Payables Turnover Ratio'] = data['Volume'] / (data['Close'] - data['Open'])
    data['Cash Conversion Cycle'] = (data['Close'] - data['Open']) / data['Volume']
    data['Debt Service Coverage Ratio'] = data['Adj Close'] / (data['Estimated Debt Volume'] * 0.05)
    data['Return on Invested Capital (ROIC)'] = (data['Close'] - data['Open']) / (data['Adj Close'] + data['Estimated Debt Volume'])
    data['Return on Common Equity (ROCE)'] = (data['Close'] - data['Open']) / data['Adj Close']
    data['Gross Margin Ratio'] = (data['Close'] - data['Open']) / data['Volume']
    data['Operating Margin Ratio'] = (data['Close'] - data['Open']) / data['Volume']
    data['Net Profit Margin Ratio'] = (data['Close'] - data['Open']) / data['Volume']
    data['Debt to Assets Ratio'] = data['Estimated Debt Volume'] / data['Asset Turnover Ratio']
    data['Equity Ratio'] = data['Volume'] / data['Asset Turnover Ratio']
    data['Financial Leverage Ratio'] = data['Asset Turnover Ratio'] / data['Volume']
    data['Proprietary Ratio'] = data['Volume'] / data['Asset Turnover Ratio']
    data['Capital Gearing Ratio'] = data['Estimated Debt Volume'] / data['Volume']
    data['Interest Coverage Ratio'] = data['EBIT'] / (data['Estimated Debt Volume'] * data['Interest Rate'])
    data['DSCR'] = (data['Adj Close'] * data['Volume']) / (data['Estimated Debt Volume'])
    data['Gross Profit Ratio'] = (data['Adj Close'] * data['Volume']) - (data['Close'] * data['Volume']) / (data['Adj Close'] * data['Volume'])
    data['Net Profit Ratio'] = (data['Close'] * data['Volume']) * data['Corporate Tax'] / (data['Adj Close'] * data['Volume'])
    data['ROI'] = (data['Close'] * data['Volume']) * data['Corporate Tax'] / data['High']
    data['EBITDA Margin'] = data['EBIT'] / (data['Adj Close'] * data['Volume'])
    data['Fixed Asset Turnover Ratio'] = (data['Adj Close'] * data['Volume']) / data['Volume'] * (data['Open'] + data['Close']) / 2
    data['Capital Turnover Ratio'] = (data['Adj Close'] * data['Volume']) / (data['Volume'] + data['Estimated Debt Volume'])
    return data

# Apply metric calculations
if acquirer_data is not None and acquiree_data is not None:
    acquirer_data = calculate_metrics(acquirer_data)
    acquiree_data = calculate_metrics(acquiree_data)
    
    # Plot each column
    for column in acquirer_data.columns:
        if column != 'Date':
            plt.figure(figsize=(10, 6))
            plt.plot(acquirer_data.index, acquirer_data[column], label='Acquirer', color='red')
            plt.plot(acquiree_data.index, acquiree_data[column], label='Acquiree', color='blue')
            plt.title(column)
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.legend()
            st.pyplot(plt)
    
    # Calculate compatibility score
    compatibility_score = 0
    metrics = ['Debt-to-Equity Ratio', 'Current Ratio', 'Interest Coverage Ratio', 'Debt-to-Capital Ratio', 'Price-to-Earnings Ratio', 'Price-to-Book Ratio', 'Return on Equity (ROE)', 'Return on Assets (ROA)', 'Earnings Yield', 'Dividend Yield', 'Price-to-Sales Ratio', 'Enterprise Value-to-EBITDA Ratio', 'Inventory Turnover Ratio', 'Receivables Turnover Ratio', 'Payables Turnover Ratio', 'Cash Conversion Cycle', 'Debt Service Coverage Ratio', 'Return on Invested Capital (ROIC)', 'Return on Common Equity (ROCE)', 'Gross Margin Ratio', 'Operating Margin Ratio', 'Net Profit Margin Ratio', 'Debt to Assets Ratio', 'Equity Ratio', 'Financial Leverage Ratio', 'Proprietary Ratio', 'Capital Gearing Ratio', 'Interest Coverage Ratio', 'DSCR', 'Gross Profit Ratio', 'Net Profit Ratio', 'ROI', 'EBITDA Margin', 'Fixed Asset Turnover Ratio', 'Capital Turnover Ratio']
    
    for metric in metrics:
        if metric in acquirer_data.columns and metric in acquiree_data.columns:
            correlation = acquirer_data[metric].corr(acquiree_data[metric])
            if correlation > 0.75:
                compatibility_score += 1
            elif correlation < -0.75:
                compatibility_score -= 1
    
    st.write(f"Compatibility Score: {compatibility_score}")

# Fetch and display news data using gazpacho
def fetch_news(ticker):
    base_url = "https://finance.yahoo.com/quote/"
    url = base_url + ticker
    html = gazpacho.get(url).text
    news_items = gazpacho.Soup(html).find("li", {"class": "js-stream-content"})
    news_data = []
    for item in news_items:
        title = item.find("h3").text
        summary = item.find("p").text
        news_data.append({"Title": title, "Summary": summary})
    return news_data

# Display news data for both acquirer and acquiree
acquirer_news = fetch_news(acquirer_ticker)
acquiree_news = fetch_news(acquiree_ticker)

st.subheader(f"{acquirer_ticker} News")
for news in acquirer_news:
    st.write(f"**Title:** {news['Title']}")
    st.write(f"**Summary:** {news['Summary']}")
    st.write("")

st.subheader(f"{acquiree_ticker} News")
for news in acquiree_news:
    st.write(f"**Title:** {news['Title']}")
    st.write(f"**Summary:** {news['Summary']}")
    st.write("")

# Perform NLP sentiment analysis
def sentiment_analysis(news_data):
    sia = SentimentIntensityAnalyzer()
    for news in news_data:
        score = sia.polarity_scores(news['Summary'])
        news['Sentiment'] = score
    return news_data

# Analyze sentiment for news data
acquirer_news_sentiment = sentiment_analysis(acquirer_news)
acquiree_news_sentiment = sentiment_analysis(acquiree_news)

st.subheader(f"{acquirer_ticker} News Sentiment Analysis")
for news in acquirer_news_sentiment:
    st.write(f"**Title:** {news['Title']}")
    st.write(f"**Summary:** {news['Summary']}")
    st.write(f"**Sentiment:** {news['Sentiment']}")
    st.write("")

st.subheader(f"{acquiree_ticker} News Sentiment Analysis")
for news in acquiree_news_sentiment:
    st.write(f"**Title:** {news['Title']}")
    st.write(f"**Summary:** {news['Summary']}")
    st.write(f"**Sentiment:** {news['Sentiment']}")
    st.write("")
