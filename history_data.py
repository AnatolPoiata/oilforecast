import time
import pandas as pd
import requests
import json


def ohlc(code: str, timeframe: str, timerange: str) -> pd.DataFrame:
    """retrieve ohlc data

        import marketwatchdata as mw

        df = mw.ohlc('STOCK/US/XNAS/AAPL', 'P1D', 'P1Y')
        print(df)

                       open      high      low   close      volume
        date
        2022-07-21  154.500  155.5700  151.940  155.35  65086641.0
        2022-07-22  155.390  156.2800  153.410  154.09  66675406.0
        2022-07-25  154.010  155.0400  152.280  152.95  53623953.0
        2022-07-26  152.265  153.0850  150.800  151.60  55138688.0
        2022-07-27  152.580  157.3300  152.160  156.79  78620688.0
        ...             ...       ...      ...     ...         ...
        2023-07-14  190.230  191.1799  189.630  190.69  41616238.0
        2023-07-17  191.900  194.3200  191.810  193.99  50520160.0
        2023-07-18  193.350  194.3300  192.415  193.73  48353770.0
        2023-07-19  193.100  198.2300  192.650  195.10  80507320.0
        2023-07-20  195.090  196.4700  192.495  193.13  59581196.0

    Args:
        code (str): code of an instrument, eg: STOCK/US/XNAS/AAPL | TMUBMUSD10Y
        timeframe (str): P1D | P1W
        timerange (str): P1M P1Y P2Y All

    Returns:
        pandas.dataframe
    """
    req_url = (
        f'https://api-secure.wsj.net/api/michelangelo/timeseries/history?json={{"Step":"{timeframe}","TimeFrame":"{timerange}", '
        '"EntitlementToken":"cecc4267a0194af89ca343805a3e57af","IncludeMockTick":true,"FilterNullSlots":false,'
        '"FilterClosedPoints":true,"IncludeClosedSlots":false,"IncludeOfficialClose":true,"InjectOpen":false,'
        '"ShowPreMarket":false,"ShowAfterHours":false,"UseExtendedTimeFrame":false,"WantPriorClose":true,'
        '"IncludeCurrentQuotes":false,"ResetTodaysAfterHoursPercentChange":false,'
        f'"Series":[{{"Key":"{code}","Dialect":"Charting","Kind":"Ticker","SeriesId":"s1",'
        '"DataTypes":["Open", "High", "Low", "Last", "Volume"]}]}&ckey=cecc4267a0'
    )

    r = requests.get(
        req_url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
            "Content-Type": "application/json, text/javascript, */*; q=0.01",
            "Dylan2010.EntitlementToken": "cecc4267a0194af89ca343805a3e57af",
        },
    )

    if (r.status_code >= 400):
        return pd.DataFrame()

    try:
        datajson = json.loads(r.content)
    except:
        return pd.DataFrame()

    times = datajson['TimeInfo']['Ticks']

    # Stock Prices
    data = datajson['Series'][0]['DataPoints']

    temp_df = pd.DataFrame(data)
    temp_df['date'] = times
    temp_df.columns = ["open", "high", "low", "close", 'volume', 'date']

    temp_df["date"] = pd.to_datetime(temp_df["date"], unit='ms')
    temp_df.set_index(["date"], inplace=True)
    temp_df.dropna(inplace=True)

    return temp_df


if __name__ == "__main__":
    df = ohlc('cl.1', 'P1D', 'P3Y')

    df['date'] = df.index

    if (df.shape[0] > 0):
        df.to_csv('historical_data.csv', date_format="%Y-%m-%d", index=False)
