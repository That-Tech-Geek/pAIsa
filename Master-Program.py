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
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from newspaper import Article
    from newspaper.article import ArticleException
    
    # Set up the Streamlit app
    st.title("Financial Data Analysis and NLP News Sentiment")
    
    # Get user input for ticker symbols
    st.header("Enter Ticker Symbols")
    acquirer_ticker = st.text_input("Acquirer Ticker")
    acquiree_ticker = st.text_input("Acquiree Ticker")
    
    # Download financial data from Yahoo Finance
    if st.button("Download Financial Data"):
        acquirer_info = yf.Ticker(acquirer_ticker).info
        acquiree_info = yf.Ticker(acquiree_ticker).info
        
        # Display stock information as a dataframe
        country = acquirer_info.get('country', 'N/A')
        sector = acquirer_info.get('sector', 'N/A')
        industry = acquirer_info.get('industry', 'N/A')
        market_cap = acquirer_info.get('marketCap', 'N/A')
        ent_value = acquirer_info.get('enterpriseValue', 'N/A')
        employees = acquirer_info.get('fullTimeEmployees', 'N/A')
        
        stock_info = [
            ("Stock Info", "Value"),
            ("Country", country),
            ("Sector", sector),
            ("Industry", industry),
            ("Market Cap", format_value(market_cap)),
            ("Enterprise Value", format_value(ent_value)),
            ("Employees", employees)
        ]
        
        df = pd.DataFrame(stock_info[1:], columns=stock_info[0])
        st.dataframe(df, width=400, hide_index=True)
        
        # Display price information as a dataframe
        current_price = acquirer_info.get('currentPrice', 'N/A')
        prev_close = acquirer_info.get('previousClose', 'N/A')
        day_high = acquirer_info.get('dayHigh', 'N/A')
        day_low = acquirer_info.get('dayLow', 'N/A')
        ft_week_high = acquirer_info.get('fiftyTwoWeekHigh', 'N/A')
        ft_week_low = acquirer_info.get('fiftyTwoWeekLow', 'N/A')
        
        price_info = [
            ("Price Info", "Value"),
            ("Current Price", f"${current_price:.2f}"),
            ("Previous Close", f"${prev_close:.2f}"),
            ("Day High", f"${day_high:.2f}"),
            ("Day Low", f"${day_low:.2f}"),
            ("52 Week High", f"${ft_week_high:.2f}"),
            ("52 Week Low", f"${ft_week_low:.2f}")
        ]
        
        df = pd.DataFrame(price_info[1:], columns=price_info[0])
        st.dataframe(df, width=400, hide_index=True)
        
        # Display business metrics as a dataframe
        forward_eps = acquirer_info.get('forwardEps', 'N/A')
        forward_pe = acquirer_info.get('forwardPE', 'N/A')
        peg_ratio = acquirer_info.get('pegRatio', 'N/A')
        dividend_rate = acquirer_info.get('dividendRate', 'N/A')
        dividend_yield = acquirer_info.get('dividendYield', 'N/A')
        recommendation = acquirer_info.get('recommendationKey', 'N/A')
        
        biz_metrics = [
            ("Business Metrics", "Value"),
            ("EPS (FWD)", f"{forward_eps:.2f}"),
            ("P/E (FWD)", f"{forward_pe:.2f}"),
            ("PEG Ratio", f"{peg_ratio:.2f}"),
            ("Div Rate (FWD)", f"${dividend_rate:.2f}"),
            ("Div Yield (FWD)", f"{dividend_yield * 100:.2f}%"),
            ("Recommendation", recommendation.capitalize())
        ]
        
        df = pd.DataFrame(biz_metrics[1:], columns=biz_metrics[0])
        st.dataframe(df, width=400, hide_index=True)
        
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
