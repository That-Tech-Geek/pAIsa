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
    # Input fields
    acquirer_ticker = st.text_input("Enter acquirer's stock ticker:")
    acquirer_exchange = st.selectbox("Select acquirer's exchange:", list(exchange_suffixes.keys()))
    
    acquiree_ticker = st.text_input("Enter acquiree's stock ticker:")
    acquiree_exchange = st.selectbox("Select acquiree's exchange:", list(exchange_suffixes.keys()))
    
    date_range = st.selectbox("Select date range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
    
    if date_range == "1d":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=1)
        end_date = pd.to_datetime('today')
    elif date_range == "5d":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=5)
        end_date = pd.to_datetime('today')
    elif date_range == "1mo":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=30)
        end_date = pd.to_datetime('today')
    elif date_range == "3mo":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=90)
        end_date = pd.to_datetime('today')
    elif date_range == "6mo":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=180)
        end_date = pd.to_datetime('today')
    elif date_range == "1y":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=365)
        end_date = pd.to_datetime('today')
    elif date_range == "2y":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=730)
        end_date = pd.to_datetime('today')
    elif date_range == "5y":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=1825)
        end_date = pd.to_datetime('today')
    elif date_range == "10y":
        start_date = pd.to_datetime('today') - pd.Timedelta(days=3650)
        end_date = pd.to_datetime('today')
    elif date_range == "ytd":
        start_date = pd.to_datetime(f'{pd.to_datetime("today").year}-01-01')
        end_date = pd.to_datetime('today')
    elif date_range == "max":
        start_date = pd.to_datetime('1924-01-01')
        end_date = pd.to_datetime('today')
    
    if acquirer_ticker and acquirer_exchange and acquiree_ticker and acquiree_exchange and start_date and end_date:
            acquirer_ticker_with_suffix = acquirer_ticker + exchange_suffixes[acquirer_exchange]
            acquiree_ticker_with_suffix = acquiree_ticker + exchange_suffixes[acquiree_exchange]
            
            def fetch_data(ticker, start, end):
                data = yf.download(ticker, start=start, end=end, progress=False)
                return data
        
        acquirer_data = fetch_data(acquirer_ticker_with_suffix, start=start_date, end=end_date)
        acquiree_data = fetch_data(acquiree_ticker_with_suffix, start=start_date, end=end_date)
    else:
        st.write("Enter all the data!")
        # Calculate estimated debt volume and other metrics
        acquirer_data['Estimated Debt Volume'] = (acquirer_data['Close'] - acquirer_data['Adj Close']) * acquirer_data['Volume']
        acquirer_data['Average Total Assets'] = acquirer_data['Adj Close'] * acquirer_data['Volume']
        acquirer_data['Asset Turnover Ratio'] = acquirer_data['Volume'] / acquirer_data['Average Total Assets']
        acquirer_data['EBIT'] = (acquirer_data['Volume'] * acquirer_data['Close']) - (acquirer_data['Volume'] * acquirer_data['Close']) - ((acquirer_data['Volume'] * acquirer_data['Close']) * (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Volume'])
        acquirer_data['Interest Rate'] = 0.08
        acquirer_data['Corporate Tax'] = 0.235
        
        # Calculate various ratios
        acquirer_data['Debt-to-Equity Ratio'] = acquirer_data['Estimated Debt Volume'] / acquirer_data['Adj Close']
        acquirer_data['Current Ratio'] = acquirer_data['Adj Close'] / acquirer_data['Estimated Debt Volume']
        acquirer_data['Interest Coverage Ratio'] = acquirer_data['Adj Close'] / (acquirer_data['Estimated Debt Volume'] * 0.05)
        acquirer_data['Debt-to-Capital Ratio'] = acquirer_data['Estimated Debt Volume'] / (acquirer_data['Adj Close'] + acquirer_data['Estimated Debt Volume'])
        acquirer_data['Price-to-Earnings Ratio'] = acquirer_data['Close'] / acquirer_data['Adj Close']
        acquirer_data['Price-to-Book Ratio'] = acquirer_data['Close'] / acquirer_data['Adj Close']
        acquirer_data['Return on Equity (ROE)'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Adj Close']
        acquirer_data['Return on Assets (ROA)'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Volume']
        acquirer_data['Earnings Yield'] = acquirer_data['Adj Close'] / acquirer_data['Close']
        acquirer_data['Dividend Yield'] = acquirer_data['Adj Close'] / acquirer_data['Close']
        acquirer_data['Price-to-Sales Ratio'] = acquirer_data['Close'] / acquirer_data['Volume']
        acquirer_data['Enterprise Value-to-EBITDA Ratio'] = (acquirer_data['Close'] * acquirer_data['Volume']) / (acquirer_data['Adj Close'] * 0.05)
        acquirer_data['Asset Turnover Ratio'] = acquirer_data['Volume'] / acquirer_data['Adj Close']
        acquirer_data['Inventory Turnover Ratio'] = acquirer_data['Volume'] / (acquirer_data['Close'] - acquirer_data['Open'])
        acquirer_data['Receivables Turnover Ratio'] = acquirer_data['Volume'] / (acquirer_data['Close'] - acquirer_data['Open'])
        acquirer_data['Payables Turnover Ratio'] = acquirer_data['Volume'] / (acquirer_data['Close'] - acquirer_data['Open'])
        acquirer_data['Cash Conversion Cycle'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Volume']
        acquirer_data['Interest Coverage Ratio'] = acquirer_data['Adj Close'] / (acquirer_data['Estimated Debt Volume'] * 0.05)
        acquirer_data['Debt Service Coverage Ratio'] = acquirer_data['Adj Close'] / (acquirer_data['Estimated Debt Volume'] * 0.05)
        acquirer_data['Return on Invested Capital (ROIC)'] = (acquirer_data['Close'] - acquirer_data['Open']) / (acquirer_data['Adj Close'] + acquirer_data['Estimated Debt Volume'])
        acquirer_data['Return on Common Equity (ROCE)'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Adj Close']
        acquirer_data['Gross Margin Ratio'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Volume']
        acquirer_data['Operating Margin Ratio'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Volume']
        acquirer_data['Net Profit Margin Ratio'] = (acquirer_data['Close'] - acquirer_data['Open']) / acquirer_data['Volume']
        acquirer_data['Debt to Assets Ratio'] = acquirer_data['Estimated Debt Volume'] / acquirer_data['Asset Turnover Ratio']
        acquirer_data['Equity Ratio'] = acquirer_data['Volume'] / acquirer_data['Asset Turnover Ratio']
        acquirer_data['Financial Leverage Ratio'] = acquirer_data['Asset Turnover Ratio'] / acquirer_data['Volume']
        acquirer_data['Proprietary Ratio'] = acquirer_data['Volume'] / acquirer_data['Asset Turnover Ratio']
        acquirer_data['Capital Gearing Ratio'] = acquirer_data['Estimated Debt Volume'] / acquirer_data['Volume']
        acquirer_data['Interest Coverage Ratio'] = acquirer_data['EBIT'] / (acquirer_data['Estimated Debt Volume'] * acquirer_data['Interest Rate'])
        acquirer_data['DSCR'] = (acquirer_data['Adj Close'] * acquirer_data['Volume']) / (acquirer_data['Estimated Debt Volume'])
        acquirer_data['Gross Profit Ratio'] = (acquirer_data['Adj Close'] * acquirer_data['Volume']) - (acquirer_data['Close'] * acquirer_data['Volume']) / (acquirer_data['Adj Close'] * acquirer_data['Volume'])
        acquirer_data['Net Profit Ratio'] = (acquirer_data['Close'] * acquirer_data['Volume']) * acquirer_data['Corporate Tax'] / (acquirer_data['Adj Close'] * acquirer_data['Volume'])
        acquirer_data['ROI'] = (acquirer_data['Close'] * acquirer_data['Volume']) * acquirer_data['Corporate Tax'] / acquirer_data['High']
        acquirer_data['EBITDA Margin'] = acquirer_data['EBIT'] / (acquirer_data['Adj Close'] * acquirer_data['Volume'])
        acquirer_data['Asset Turnover Ratio'] = (acquirer_data['Adj Close'] * acquirer_data['Volume']) / acquirer_data['Asset Turnover Ratio']
        acquirer_data['Fixed Asset Turnover Ratio'] = (acquirer_data['Adj Close'] * acquirer_data['Volume']) / acquirer_data['Volume'] * (acquirer_data['Open'] + acquirer_data['Close']) / 2
        acquirer_data['Capital Turnover Ratio'] = (acquirer_data['Adj Close'] * acquirer_data['Volume']) / (acquirer_data['Volume'] + acquirer_data['Estimated Debt Volume'])
        
        # Repeat the same calculations for the acquiree data
        acquiree_data['Estimated Debt Volume'] = (acquiree_data['Close'] - acquiree_data['Adj Close']) * acquiree_data['Volume']
        acquiree_data['Average Total Assets'] = acquiree_data['Adj Close'] * acquiree_data['Volume']
        acquiree_data['Asset Turnover Ratio'] = acquiree_data['Volume'] / acquiree_data['Average Total Assets']
        acquiree_data['EBIT'] = (acquiree_data['Volume'] * acquiree_data['Close']) - (acquiree_data['Volume'] * acquiree_data['Close']) - ((acquiree_data['Volume'] * acquiree_data['Close']) * (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Volume'])
        acquiree_data['Interest Rate'] = 0.08
        acquiree_data['Corporate Tax'] = 0.235
        
        # Calculate various ratios
        acquiree_data['Debt-to-Equity Ratio'] = acquiree_data['Estimated Debt Volume'] / acquiree_data['Adj Close']
        acquiree_data['Current Ratio'] = acquiree_data['Adj Close'] / acquiree_data['Estimated Debt Volume']
        acquiree_data['Interest Coverage Ratio'] = acquiree_data['Adj Close'] / (acquiree_data['Estimated Debt Volume'] * 0.05)
        acquiree_data['Debt-to-Capital Ratio'] = acquiree_data['Estimated Debt Volume'] / (acquiree_data['Adj Close'] + acquiree_data['Estimated Debt Volume'])
        acquiree_data['Price-to-Earnings Ratio'] = acquiree_data['Close'] / acquiree_data['Adj Close']
        acquiree_data['Price-to-Book Ratio'] = acquiree_data['Close'] / acquiree_data['Adj Close']
        acquiree_data['Return on Equity (ROE)'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Adj Close']
        acquiree_data['Return on Assets (ROA)'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Volume']
        acquiree_data['Earnings Yield'] = acquiree_data['Adj Close'] / acquiree_data['Close']
        acquiree_data['Dividend Yield'] = acquiree_data['Adj Close'] / acquiree_data['Close']
        acquiree_data['Price-to-Sales Ratio'] = acquiree_data['Close'] / acquiree_data['Volume']
        acquiree_data['Enterprise Value-to-EBITDA Ratio'] = (acquiree_data['Close'] * acquiree_data['Volume']) / (acquiree_data['Adj Close'] * 0.05)
        acquiree_data['Asset Turnover Ratio'] = acquiree_data['Volume'] / acquiree_data['Adj Close']
        acquiree_data['Inventory Turnover Ratio'] = acquiree_data['Volume'] / (acquiree_data['Close'] - acquiree_data['Open'])
        acquiree_data['Receivables Turnover Ratio'] = acquiree_data['Volume'] / (acquiree_data['Close'] - acquiree_data['Open'])
        acquiree_data['Payables Turnover Ratio'] = acquiree_data['Volume'] / (acquiree_data['Close'] - acquiree_data['Open'])
        acquiree_data['Cash Conversion Cycle'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Volume']
        acquiree_data['Interest Coverage Ratio'] = acquiree_data['Adj Close'] / (acquiree_data['Estimated Debt Volume'] * 0.05)
        acquiree_data['Debt Service Coverage Ratio'] = acquiree_data['Adj Close'] / (acquiree_data['Estimated Debt Volume'] * 0.05)
        acquiree_data['Return on Invested Capital (ROIC)'] = (acquiree_data['Close'] - acquiree_data['Open']) / (acquiree_data['Adj Close'] + acquiree_data['Estimated Debt Volume'])
        acquiree_data['Return on Common Equity (ROCE)'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Adj Close']
        acquiree_data['Gross Margin Ratio'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Volume']
        acquiree_data['Operating Margin Ratio'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Volume']
        acquiree_data['Net Profit Margin Ratio'] = (acquiree_data['Close'] - acquiree_data['Open']) / acquiree_data['Volume']
        acquiree_data['Debt to Assets Ratio'] = acquiree_data['Estimated Debt Volume'] / acquiree_data['Asset Turnover Ratio']
        acquiree_data['Equity Ratio'] = acquiree_data['Volume'] / acquiree_data['Asset Turnover Ratio']
        acquiree_data['Financial Leverage Ratio'] = acquiree_data['Asset Turnover Ratio'] / acquiree_data['Volume']
        acquiree_data['Proprietary Ratio'] = acquiree_data['Volume'] / acquiree_data['Asset Turnover Ratio']
        acquiree_data['Capital Gearing Ratio'] = acquiree_data['Estimated Debt Volume'] / acquiree_data['Volume']
        acquiree_data['Interest Coverage Ratio'] = acquiree_data['EBIT'] / (acquiree_data['Estimated Debt Volume'] * acquiree_data['Interest Rate'])
        acquiree_data['DSCR'] = (acquiree_data['Adj Close'] * acquiree_data['Volume']) / (acquiree_data['Estimated Debt Volume'])
        acquiree_data['Gross Profit Ratio'] = (acquiree_data['Adj Close'] * acquiree_data['Volume']) - (acquiree_data['Close'] * acquiree_data['Volume']) / (acquiree_data['Adj Close'] * acquiree_data['Volume'])
        acquiree_data['Net Profit Ratio'] = (acquiree_data['Close'] * acquiree_data['Volume']) * acquiree_data['Corporate Tax'] / (acquiree_data['Adj Close'] * acquiree_data['Volume'])
        acquiree_data['ROI'] = (acquiree_data['Close'] * acquiree_data['Volume']) * acquiree_data['Corporate Tax'] / acquiree_data['High']
        acquiree_data['EBITDA Margin'] = acquiree_data['EBIT'] / (acquiree_data['Adj Close'] * acquiree_data['Volume'])
        acquiree_data['Asset Turnover Ratio'] = (acquiree_data['Adj Close'] * acquiree_data['Volume']) / acquiree_data['Asset Turnover Ratio']
        acquiree_data['Fixed Asset Turnover Ratio'] = (acquiree_data['Adj Close'] * acquiree_data['Volume']) / acquiree_data['Volume'] * (acquiree_data['Open'] + acquiree_data['Close']) / 2
        acquiree_data['Capital Turnover Ratio'] = (acquiree_data['Adj Close'] * acquiree_data['Volume']) / (acquiree_data['Volume'] + acquiree_data['Estimated Debt Volume'])
       
        # Plot comparison of metrics
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.lineplot(x=acquirer_data.index, y=acquirer_data['Debt-to-Equity Ratio'], ax=ax[0], label=acquirer_ticker)
        sns.lineplot(x=acquiree_data.index, y=acquiree_data['Debt-to-Equity Ratio'], ax=ax[0], label=acquiree_ticker)
        ax[0].set_title("Debt-to-Equity Ratio")
        
        sns.lineplot(x=acquirer_data.index, y=acquirer_data['Current Ratio'], ax=ax[1], label=acquirer_ticker)
        sns.lineplot(x=acquiree_data.index, y=acquiree_data['Current Ratio'], ax=ax[1], label=acquiree_ticker)
        ax[1].set_title("Current Ratio")
        
        st.pyplot(fig)
        
        # Calculate compatibility score
        compatibility_score = 0
        for metric in ['Debt-to-Equity Ratio', 'Current Ratio', 'Interest Coverage Ratio',...]:
            acquirer_metric = acquirer_data[metric].mean()
            acquiree_metric = acquiree_data[metric].mean()
            compatibility_score += abs(acquirer_metric - acquiree_metric)
        
        st.write("Compatibility Score:", compatibility_score)
                
        try:
            html = gazpacho.get(news_url)
            soup = gazpacho.Soup(html)
            article_text = ""
            
            # Find the article content using a CSS selector
            for paragraph in soup.find("p", mode="all"):
                article_text += paragraph.text + " "
            
            # Tokenize the text
            tokens = word_tokenize(article_text)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
            
            # Calculate sentiment score
            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(' '.join(tokens))
            
            # Display sentiment score
            st.write("Sentiment Score:")
            st.write(f"Positive: {sentiment_score['pos']:.2f}")
            st.write(f"Negative: {sentiment_score['neg']:.2f}")
            st.write(f"Neutral: {sentiment_score['neu']:.2f}")
            st.write(f"Compound: {sentiment_score['compound']:.2f}")
            
            # Determine sentiment
            if sentiment_score['compound'] > 0.05:
                sentiment = "Positive"
            elif sentiment_score['compound'] < -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            st.write(f"Overall Sentiment: {sentiment}")
        except Exception as e:
            st.write("Error:", e)
