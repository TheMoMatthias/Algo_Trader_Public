import pandas as pd
import numpy as np
import san 
import datetime as dt


san.ApiConfig.api_key = "3yqjmkzpt46wdswa_rt62wevhab5dptkz"

import san


######################################################## san functions    #################################

san.api_calls_made()
san.api_calls_remaining()
san.is_rate_limit_exception()
san.rate_limit_time_left()

######################################################## basic functions   #################################


test_1 = san.get_many(
  "price_usd",
  slugs=["bitcoin", "ethereum", "tether"],
  from_date="2017-01-01",
  to_date="2018-01-05",
  interval="1d"
)


test_2 = san.get(
  "dev_activity",
  selector={"organization": "google"},
  from_date="2022-01-01",
  to_date="2022-01-05",
  interval="1d"
)

test_3 = san.get(
  "amount_in_top_holders",
  selector={"slug": "santiment", "holdersCount": 10},
  from_date="2019-01-01",
  to_date="2020-01-05",
  interval="1d"
)




overview = san.get('projects/all')

# overview_2 = san.get('metrics/all')'

# test_4 = san.get(
#   "total_trade_volume_by_dex",
#   selector={"slug": "ethereum", "label": "decentralized_exchange", "owner": "UniswapV2"},
#   from_date="2022-01-01",
#   to_date="2022-01-05",
#   interval="1d"
# )

test_4 = san.get(
    "token_top_transactions",
    slug="santiment",
    from_date="2019-04-18",
    to_date="2019-04-30",
    limit=5
)


test_10 = san.get('daily_trading_volume_usd', 
        slug='ethereum',
        from_date ="2019-04-18",
        to_date="2019-04-30",
        interval="1d")

overview 


import san
import pandas as pd


##############################################################    query via graph if not available   ###############################################
result = san.graphql.execute_gql("""
{
  getMetric(metric: "price_usd") {
    timeseriesDataPerSlug(
      selector: {slugs: ["ethereum", "bitcoin"]}
      from: "2022-05-05T00:00:00Z"
      to: "2022-05-08T00:00:00Z"
      interval: "1d") {
        datetime
        data{
          value
          slug
        }
    }
  }
}
""")

# Define variables
variables = {
    "metricName": "price_usd",
    "slugList": ["ethereum", "bitcoin"],
    "startDate": "2022-05-05T00:00:00Z",
    "endDate": "2022-05-08T00:00:00Z",
    "intervalValue": "1d",
}

# transformed using placeholders
result = """
{
  getMetric(metric: $metricName) {
    timeseriesDataPerSlug(
      selector: {slugs: $slugList}
      from: $startDate
      to: $endDate
      interval: $intervalValue
    ) {
      datetime
      data {
        value
        slug
      }
    }
  }
}
"""

data = result['getMetric']['timeseriesDataPerSlug']
rows = []
for datetime_point in data:
    row = {'datetime': datetime_point['datetime']}
    for slug_data in datetime_point['data']:
        row[slug_data['slug']] = slug_data['value']
    rows.append(row)

df = pd.DataFrame(rows)
df.set_index('datetime', inplace=True)




######################################################## more advanced version with variable query parameters #################################


def backtest(self, metricName, slugList, startDate, endDate, intervalValue):
        # Define the GraphQL query with placeholders
        query = san.graphql.execute_gql("""
        {{
            getMetric(metric: "{metricName}") {{
                timeseriesDataPerSlug(
                selector: {{slugs: {slugList}}}
                from: "{startDate}"
                to: "{endDate}"
                interval: "{intervalValue}"
                ) {{
                    datetime
                    data {{
                        value
                        slug
                    }}
                }}
            }}
        }}
        """.format(
            metricName=metricName,
            slugList=json.dumps(slugList),  # Convert slugList to a JSON string
            startDate=startDate,
            endDate=endDate,
            intervalValue=intervalValue))

        data = query['getMetric']['timeseriesDataPerSlug']
        rows = []
        for datetime_point in data:
            row = {'datetime': datetime_point['datetime']}
            for slug_data in datetime_point['data']:
                row[slug_data['slug']] = slug_data['value']
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index('datetime', inplace=True)

        print('test')

    def backtest_the_backtest(self):
        inputs = [
        {
            "metricName": "amount_in_top_holders",
            "slugList": ["ethereum", "bitcoin", "quant"],
            "startDate": "2022-05-05T00:00:00Z",
            "endDate": "2022-05-08T00:00:00Z",
            "intervalValue": "1d",
        },
        # Add more dictionaries with different inputs as needed
        ]

        for input_params in inputs:
            self.backtest(**input_params)

########################################################   Transform  query to average   ######################################################

san.get(
  "price_usd",
  slug="santiment",
  from_date="2020-06-01",
  to_date="2021-06-05",
  interval="1d",
  transform={"type": "moving_average", "moving_average_base": 100},
  aggregation="LAST"
)


# other transformers: consecutive_differences,


######################################################    get overview of all metrics    ################################################
all_metrics = san.available_metrics()
all_metrics_btc = san.available_metrics_for_slug("bitcoin")



#####################################################    see since when metric is available   ################################

san.available_metric_for_slug_since(metric="daily_active_addresses", slug="santiment")

################################################   see which assets are available for a metric    ########################################
san.metadata(
    "nvt",
    arr=["availableSlugs", "defaultAggregation", "humanReadableName", "isAccessible", "isRestricted", "restrictedFrom", "restrictedTo"]
)

"""
result: {"availableSlugs": ["0chain", "0x", "0xbtc", "0xcert", "1sg", ...],
"defaultAggregation": "AVG", "humanReadableName": "NVT (Using Circulation)", "isAccessible": True, "isRestricted": True, "restrictedFrom": "2020-03-21T08:44:14Z", "restrictedTo": "2020-06-17T08:44:14Z"}

"""


######################################################################################################################################################
# batching data queries 

from san import AsyncBatch

batch = AsyncBatch()

batch.get(
    "daily_active_addresses",
    slug="santiment",
    from_date="2018-06-01",
    to_date="2018-06-05",
    interval="1d"
)
batch.get_many(
    "daily_active_addresses",
    slugs=["bitcoin", "ethereum"],
    from_date="2018-06-01",
    to_date="2018-06-05",
    interval="1d"
)
[daa, daa_many] = batch.execute(max_workers=10)


########################################################## Limit requests #######################################################################


import time
import san

try:
  san.get(
    "price_usd",
    slug="santiment",
    from_date="utc_now-30d",
    to_date="utc_now",
    interval="1d"
  )
except Exception as e:
  if san.is_rate_limit_exception(e):
    rate_limit_seconds = san.rate_limit_time_left(e)
    print(f"Will sleep for {rate_limit_seconds}")
    time.sleep(rate_limit_seconds)

...

calls_by_day = san.api_calls_made()
calls_remaining = san.api_calls_remaining()


##########################################################   get complexity requests  ###############################################################
complexity = san.metric_complexity(
    metric="price_usd",
    from_date="2017-01-01",
    to_date="2023-10-13",
    interval="5m"
)

if complexity > 50000:
   adjustment_factor = np.round(complexity/50000)



#NOTE: 50000 complexity score is maximum allowed

##################################################################  Search for topics ################################################

def topic_search(idx, **kwargs):
    kwargs = sgh.transform_query_args('topic_search', **kwargs)
    query_str = ("""
    query_{idx}: topicSearch (
        source: {source},
        searchText: \"{search_text}\",
        from: \"{from_date}\",
        to: \"{to_date}\",
        interval: \"{interval}\"
    ){{
    """ + ' '.join(kwargs['return_fields']) + """
    }}
    """).format(
        idx=idx,
        **kwargs
    )



def news(idx, tag, **kwargs):
    print('WARNING! This metric is going to be removed in version 0.8.0')
    kwargs = sgh.transform_query_args('news', **kwargs)

    query_str = ("""
    query_{idx}: news(
        tag: \"{tag}\",
        from: \"{from_date}\",
        to: \"{to_date}\",
        size: {size}
    ){{
    """ + ' '.join(kwargs['return_fields']) + '}}').format(
        idx=idx,
        tag=tag,
        **kwargs
    )


print('success')
print('success')