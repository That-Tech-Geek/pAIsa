import streamlit as st
import yfinance as yf
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gazpacho
from newspaper.article import ArticleException

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
    # Set up the Streamlit app
    st.title("Financial Data Analysis and NLP News Sentiment")
    
    # Get user input for ticker symbols
    st.header("Enter Ticker Symbols")
    acquirer_ticker = st.text_input("Acquirer Ticker")
    acquiree_ticker = st.text_input("Acquiree Ticker")
    
    # Download financial data from Yahoo Finance
    if st.button("Download Financial Data"):
        # Download financial data from Yahoo Finance
        acquirer_data = yf.download(acquirer_ticker, period="max")
        acquiree_data = yf.download(acquiree_ticker, period="max")
        
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
        
        # Now that we have calculated all the financial ratios for both the acquirer and acquiree,
        # we can perform a comparative analysis to identify potential synergies and areas of improvement.
        
        # Let's start by comparing the debt-to-equity ratios of both companies
        debt_to_equity_ratio_acquirer = acquirer_data['Debt-to-Equity Ratio'].iloc[-1]
        debt_to_equity_ratio_acquiree = acquiree_data['Debt-to-Equity Ratio'].iloc[-1]
        
        print(f"Acquirer's Debt-to-Equity Ratio: {debt_to_equity_ratio_acquirer:.2f}")
        print(f"Acquiree's Debt-to-Equity Ratio: {debt_to_equity_ratio_acquiree:.2f}")
        
        # Compare the interest coverage ratios of both companies
        interest_coverage_ratio_acquirer = acquirer_data['Interest Coverage Ratio'].iloc[-1]
        interest_coverage_ratio_acquiree = acquiree_data['Interest Coverage Ratio'].iloc[-1]
        
        print(f"Acquirer's Interest Coverage Ratio: {interest_coverage_ratio_acquirer:.2f}")
        print(f"Acquiree's Interest Coverage Ratio: {interest_coverage_ratio_acquiree:.2f}")
        
        # Compare the return on equity (ROE) of both companies
        roe_acquirer = acquirer_data['Return on Equity (ROE)'].iloc[-1]
        roe_acquiree = acquiree_data['Return on Equity (ROE)'].iloc[-1]
        
        print(f"Acquirer's ROE: {roe_acquirer:.2f}%")
        print(f"Acquiree's ROE: {roe_acquiree:.2f}%")
        
        # Compare the return on assets (ROA) of both companies
        roa_acquirer = acquirer_data['Return on Assets (ROA)'].iloc[-1]
        roa_acquiree = acquiree_data['Return on Assets (ROA)'].iloc[-1]
        
        print(f"Acquirer's ROA: {roa_acquirer:.2f}%")
        print(f"Acquiree's ROA: {roa_acquiree:.2f}%")        
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
            article_content = soup.find("div.article-body", mode="first")
            article_text = article_content.text
            
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
