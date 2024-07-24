import streamlit as st
import yfinance as yf
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from newspaper import Article
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
        acquirer_data = yf.download(acquirer_ticker, period="1y")
        acquiree_data = yf.download(acquiree_ticker, period="1y")
        
        # Calculate estimated debt volume and other metrics
        acquirer_data['Estimated Debt Volume'] = (acquirer_data['Close'] - acquirer_data['Adj Close']) * acquirer_data['Volume']
        acquiree_data['Estimated Debt Volume'] = (acquiree_data['Close'] - acquiree_data['Adj Close']) * acquiree_data['Volume']
        
        # Calculate various ratios
        acquirer_data['Debt-to-Equity Ratio'] = acquirer_data['Estimated Debt Volume'] / acquirer_data['Adj Close']
        acquiree_data['Debt-to-Equity Ratio'] = acquiree_data['Estimated Debt Volume'] / acquiree_data['Adj Close']
        
        acquirer_data['Current Ratio'] = acquirer_data['Adj Close'] / acquirer_data['Estimated Debt Volume']
        acquiree_data['Current Ratio'] = acquiree_data['Adj Close'] / acquiree_data['Estimated Debt Volume']
        
        acquirer_data['Interest Coverage Ratio'] = acquirer_data['Adj Close'] / (acquirer_data['Estimated Debt Volume'] * 0.05)
        acquiree_data['Interest Coverage Ratio'] = acquiree_data['Adj Close'] / (acquiree_data['Estimated Debt Volume'] * 0.05)
        
        #... calculate other metrics...
        
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
        
        # Analyze news articles using NLP
        st.header("News Sentiment Analysis")
        news_url = st.text_input("Enter News Article URL")
        
        if st.button("Analyze News Sentiment"):
            try:
                article = Article(news_url)
                article.download()
                article.parse()
                text = article.text
                
                # Tokenize the text
                tokens = word_tokenize(text)
                
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
            
            except ArticleException as e:
                st.error("Error parsing news article:", e)
