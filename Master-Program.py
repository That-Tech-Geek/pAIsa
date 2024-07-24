import streamlit as st
import os

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

# Check the selected service type
if service_type == "Mergers and Acquisition Advisory Services":
    # Run the Mergers and Acquisitions code
    import pandas as pd
    import numpy as np
    import streamlit as st
    import matplotlib.pyplot as plt

    # Create a Streamlit app
    st.title("Mergers and Acquisitions Compatibility Analysis")

    # Create text inputs for acquirer and acquiree tickers
    acquirer_ticker = st.text_input("Enter NSE Ticker of Acquirer:")
    acquiree_ticker = st.text_input("Enter NSE Ticker of Acquiree:")

    # Create a button to analyze the compatibility
    if st.button("Analyze"):
        # Check if both tickers are entered
        if acquirer_ticker and acquiree_ticker:
            # Load the financial data for the acquirer and acquiree
            acquirer_data = pd.read_csv(f"D:\\Datasets\\Cleaned Datasets\\{acquirer_ticker}.csv")
            acquiree_data = pd.read_csv(f"D:\\Datasets\\Cleaned Datasets\\{acquiree_ticker}.csv")
            acquirer_sentiment = pd.read_csv(f"D:\\News Analysis Results\\{acquirer_ticker}.csv")
            acquiree_sentiment = pd.read_csv(f"D:\\News Analysis Results\\{acquiree_ticker}.csv")

            # Calculate the financial metrics
            acquirer_eps = acquirer_data['EPS'].mean()
            acquiree_eps = acquiree_data['EPS'].mean()

            acquirer_pe_ratio = acquirer_data['P/E Ratio'].mean()
            acquiree_pe_ratio = acquiree_data['P/E Ratio'].mean()

            acquirer_dividend_yield = acquirer_data['Dividend Yield'].mean()
            acquiree_dividend_yield = acquiree_data['Dividend Yield'].mean()

            acquirer_roe = acquirer_data['ROE'].mean()
            acquiree_roe = acquiree_data['ROE'].mean()

            acquirer_debt_to_equity_ratio = acquirer_data['Debt-to-Equity Ratio'].mean()
            acquiree_debt_to_equity_ratio = acquiree_data['Debt-to-Equity Ratio'].mean()

            acquirer_current_ratio = acquirer_data['Current Ratio'].mean()
            acquiree_current_ratio = acquiree_data['Current Ratio'].mean()

            acquirer_interest_coverage_ratio = acquirer_data['Interest Coverage Ratio'].mean()
            acquiree_interest_coverage_ratio = acquiree_data['Interest Coverage Ratio'].mean()

            # Calculate the sentiment metrics
            if 'Positive Sentiment' in acquirer_sentiment.columns:
                acquirer_positive_sentiment = acquirer_sentiment['Positive Sentiment'].mean()

            if 'Positive Sentiment' in acquiree_sentiment.columns:
                acquiree_positive_sentiment = acquiree_sentiment['Positive Sentiment'].mean()

            if 'Negative Sentiment' in acquirer_sentiment.columns:
                acquirer_negative_sentiment = acquirer_sentiment['Negative Sentiment'].mean()

            if 'Negative Sentiment' in acquiree_sentiment.columns:
                acquiree_negative_sentiment = acquiree_sentiment['Negative Sentiment'].mean()

            if 'Neutral Sentiment' in acquirer_sentiment.columns:
                acquirer_neutral_sentiment = acquirer_sentiment['Neutral Sentiment'].mean()

            if 'Neutral Sentiment' in acquiree_sentiment.columns:
                acquiree_neutral_sentiment = acquiree_sentiment['Neutral Sentiment'].mean()

            if 'Compound Score' in acquirer_sentiment.columns:
                acquirer_compound_score = acquirer_sentiment['Compound Score'].mean()

            if 'Compound Score' in acquiree_sentiment.columns:
                acquiree_compound_score = acquiree_sentiment['Compound Score'].mean()

            # Calculate the compatibility score
            def calculate_compatibility_score(acquirer_eps, acquiree_eps, acquirer_pe_ratio, acquiree_pe_ratio, 
                                            acquirer_dividend_yield, acquiree_dividend_yield, acquirer_roe, acquiree_roe, 
                                            acquirer_debt_to_equity_ratio, acquiree_debt_to_equity_ratio, 
                                            acquirer_current_ratio, acquiree_current_ratio, 
                                            acquirer_interest_coverage_ratio, acquiree_interest_coverage_ratio, 
                                            acquirer_positive_sentiment, acquiree_positive_sentiment, 
                                            acquirer_negative_sentiment, acquiree_negative_sentiment, 
                                            acquirer_neutral_sentiment, acquiree_neutral_sentiment, 
                                            acquirer_compound_score, acquiree_compound_score):
                financial_score = (acquirer_eps / acquiree_eps + acquirer_pe_ratio / acquiree_pe_ratio + 
                                acquirer_dividend_yield / acquiree_dividend_yield + acquirer_roe / acquiree_roe + 
                                acquirer_debt_to_equity_ratio / acquiree_debt_to_equity_ratio + 
                                acquirer_current_ratio / acquiree_current_ratio + 
                                acquirer_interest_coverage_ratio / acquiree_interest_coverage_ratio) / 7
                
                sentiment_score = (acquirer_positive_sentiment / acquiree_positive_sentiment + 
                                acquirer_negative_sentiment / acquiree_negative_sentiment + 
                                acquirer_neutral_sentiment / acquiree_neutral_sentiment + 
                                acquirer_compound_score / acquiree_compound_score) / 4
                
                compatibility_score = (financial_score + sentiment_score) ** 2
                
                return compatibility_score

            compatibility_score = calculate_compatibility_score(acquirer_eps, acquiree_eps, acquirer_pe_ratio, acquiree_pe_ratio, 
                                                                acquirer_dividend_yield, acquiree_dividend_yield, acquirer_roe, acquiree_roe, 
                                                                acquirer_debt_to_equity_ratio, acquiree_debt_to_equity_ratio, 
                                                                acquirer_current_ratio, acquiree_current_ratio, 
                                                                acquirer_interest_coverage_ratio, acquiree_interest_coverage_ratio, 
                                                                acquirer_positive_sentiment, acquiree_positive_sentiment, 
                                                                acquirer_negative_sentiment, acquiree_negative_sentiment, 
                                                                acquirer_neutral_sentiment, acquiree_neutral_sentiment, 
                                                                acquirer_compound_score, acquiree_compound_score)

            # Display the compatibility score
            st.write(f"Compatibility Score: {compatibility_score:.2f}")

            # Initialize a dictionary to store the compatibility scores
            compatibility_scores = {}

            data = pd.read_csv(r"C:\Users\91891\Downloads\Directory - Sheet1 (1).csv")

            # Iterate over each company in the data
            for index, row in data.iterrows():
                acquiree_ticker = row['NSE CODE']  # Access the 'NSE CODE' column
                
                # Skip if the acquiree ticker is the same as the acquirer ticker
                if acquiree_ticker == acquirer_ticker:
                    continue
                
                # Load the financial data for the acquiree
                acquiree_data = pd.read_csv(f"D:\\Datasets\\Cleaned Datasets\\{acquiree_ticker}.csv")
                
                # Load the sentiment data for the acquiree
                acquiree_sentiment = pd.read_csv(f"D:\\News Analysis Results\\{acquiree_ticker}.csv")
                
                # Calculate the financial metrics for the acquiree
                acquiree_eps = acquiree_data['EPS'].mean()
                acquiree_pe_ratio = acquiree_data['P/E Ratio'].mean()
                acquiree_dividend_yield = acquiree_data['Dividend Yield'].mean()
                acquiree_roe = acquiree_data['ROE'].mean()
                acquiree_debt_to_equity_ratio = acquiree_data['Debt-to-Equity Ratio'].mean()
                acquiree_current_ratio = acquiree_data['Current Ratio'].mean()
                acquiree_interest_coverage_ratio = acquiree_data['Interest Coverage Ratio'].mean()
                
                # Calculate the sentiment metrics for the acquiree
                if 'Positive Sentiment' in acquiree_sentiment.columns:
                    acquiree_positive_sentiment = acquiree_sentiment['Positive Sentiment'].mean()

                if 'Negative Sentiment' in acquiree_sentiment.columns:
                    acquiree_negative_sentiment = acquiree_sentiment['Negative Sentiment'].mean()

                if 'Neutral Sentiment' in acquiree_sentiment.columns:
                    acquiree_neutral_sentiment = acquiree_sentiment['Neutral Sentiment'].mean()

                if 'Compound Score' in acquiree_sentiment.columns:
                    acquiree_compound_score = acquiree_sentiment['Compound Score'].mean()
                
                # Calculate the compatibility score
                financial_score = (acquiree_eps / acquiree_eps + acquiree_pe_ratio / acquiree_pe_ratio + 
                                acquiree_dividend_yield / acquiree_dividend_yield + acquiree_roe / acquiree_roe + 
                                acquiree_debt_to_equity_ratio / acquiree_debt_to_equity_ratio + 
                                acquiree_current_ratio / acquiree_current_ratio + 
                                acquiree_interest_coverage_ratio / acquiree_interest_coverage_ratio) / 7
                
                sentiment_score = (acquiree_positive_sentiment/ acquiree_positive_sentiment + 
                                acquiree_negative_sentiment / acquiree_negative_sentiment + 
                                acquiree_neutral_sentiment / acquiree_neutral_sentiment + 
                                acquiree_compound_score/ acquiree_compound_score) / 4
                
                compatibility_score = (financial_score + sentiment_score) ** 2
                
                # Store the compatibility score in the dictionary
                compatibility_scores[acquiree_ticker] = compatibility_score

            # Sort the compatibility scores in descending order
            sorted_scores = sorted(compatibility_scores.items(), key=lambda x: x[1], reverse=True)

            # Display the top 25 compatibility scores
            st.write("Top 25 Compatibility Scores:")
            for i, (ticker, score) in enumerate(sorted_scores[:25]):
                st.write(f"{i+1}. {ticker} - {score:.2f}")

            # Create a Streamlit title
            st.title('Stock Price Analysis')

            # Plot the 'Open' graph
            st.header('Open')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Open'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Open'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Open')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'High' graph
            st.header('High')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['High'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['High'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('High')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Low' graph
            st.header('Low')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Low'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Low'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Low')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Close' graph
            st.header('Close')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Close'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Close'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Close')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Adj Close' graph
            st.header('Adj Close')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Adj Close'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Adj Close'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Adj Close')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Volume' graph
            st.header('Volume')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Volume'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Volume'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Volume')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'EPS' graph
            st.header('EPS')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['EPS'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['EPS'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('EPS')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'P/E Ratio' graph
            st.header('P/E Ratio')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['P/E Ratio'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['P/E Ratio'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('P/E Ratio')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Dividend Yield' graph
            st.header('Dividend Yield')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Dividend Yield'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Dividend Yield'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Dividend Yield')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'ROE' graph
            st.header('ROE')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['ROE'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['ROE'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('ROE')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Debt-to-Equity Ratio' graph
            st.header('Debt-to-Equity Ratio')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Debt-to-Equity Ratio'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Debt-to-Equity Ratio'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Debt-to-Equity Ratio')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Current Ratio' graph
            st.header('Current Ratio')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Current Ratio'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Current Ratio'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Current Ratio')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            # Plot the 'Interest Coverage Ratio' graph
            st.header('Interest Coverage Ratio')
            fig, ax = plt.subplots()
            ax.plot(acquirer_data['Interest Coverage Ratio'], color='red', label=f'{acquirer_ticker}')
            ax.plot(acquiree_data['Interest Coverage Ratio'], color='blue', label=f'{acquiree_ticker}')
            ax.set_title('Interest Coverage Ratio')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)
    st.write("Mergers and Acquisitions code running")

if service_type == "Stock Data Pulling Services":
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import streamlit as st
    if __name__ == "__main__":
        st.title("Stock Market Data Analyzer")

        # Create a directory for reference
        st.write("Directory for Reference:")
        st.write("You can search for the ticker symbol in the following directory:")
        st.write("https://docs.google.com/spreadsheets/d/e/2PACX-1vQQp6LeDQCgT2YcwvQe1ocK6LKNzp1zUhraq4E0L4-W3p_Zr6wlyHdh-vaCJzGaFdSuSIcSgmGWEHZp/pubhtml")

        # Define the ticker symbol of the company
        ticker_symbol = st.text_input("Enter NSE Ticker symbol:", value="")

        if ticker_symbol:
            try:
                print("Ticker symbol:", ticker_symbol)

                # Load data from local CSV file
                file_path = f"D:/Datasets/Cleaned Datasets/{ticker_symbol}.csv"
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    st.write("Loaded data from local CSV file:")
                    st.write(data)

                    # Plot each column separately
                    columns = ['Close', 'Adj Close', 'Volume', 'EPS', 'P/E Ratio', 'Dividend Yield', 'ROE', 'Debt-to-Equity Ratio', 'Current Ratio', 'Interest Coverage Ratio']
                    for col in columns:
                        if col in data.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if col in ['Close', 'Adj Close', 'EPS', 'P/E Ratio', 'Dividend Yield', 'ROE', 'Debt-to-Equity Ratio', 'Current Ratio', 'Interest Coverage Ratio']:
                                ax.plot(data[col])
                                ax.set_title(f'{ticker_symbol} {col} Over Time')
                            elif col == 'Volume':
                                ax.bar(data.index, data[col])
                                ax.set_title(f'{ticker_symbol} {col} Over Time')
                            ax.set_xlabel('Date')
                            ax.set_ylabel(col)
                            st.pyplot(fig)
                    st.write("The numbers you see on the horizontal axis are from the index numbers of corresponding dates. Refer to the .csv file above the graphs to access the exact date you need data for.")
                else:
                    st.write("Error: No data found for the selected ticker symbol.")
            except Exception as e:
                st.write("Error:", str(e))

else:
    st.write("In development")