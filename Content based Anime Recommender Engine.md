```python
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
%matplotlib inline
```


```python
df = pd.read_csv("/Users/adarshcj/Downloads/anime.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mal_id</th>
      <th>title</th>
      <th>synopsis</th>
      <th>background</th>
      <th>aired</th>
      <th>airing</th>
      <th>duration</th>
      <th>episodes</th>
      <th>type</th>
      <th>favorites</th>
      <th>...</th>
      <th>score</th>
      <th>scored_by</th>
      <th>rating</th>
      <th>premiered</th>
      <th>genres</th>
      <th>related</th>
      <th>status</th>
      <th>licensors</th>
      <th>producers</th>
      <th>studios</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>In the year 2071, humanity has colonized sever...</td>
      <td>-</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>59968</td>
      <td>...</td>
      <td>8.77</td>
      <td>661519</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>Spring 1998</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>{'Adaptation': [{'mal_id': 173, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>Bandai Visual</td>
      <td>Sunrise</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>Another day, another bounty—such is the life o...</td>
      <td>-</td>
      <td>Sep 1, 2001</td>
      <td>0</td>
      <td>1 hr 55 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>1063</td>
      <td>...</td>
      <td>8.39</td>
      <td>168515</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>-</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>{'Parent story': [{'mal_id': 1, 'type': 'anime...</td>
      <td>Finished Airing</td>
      <td>Sony Pictures Entertainment</td>
      <td>Sunrise, Bandai Visual</td>
      <td>Bones</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>Vash the Stampede is the man with a $$60,000,0...</td>
      <td>-</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>11882</td>
      <td>...</td>
      <td>8.23</td>
      <td>288760</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Spring 1998</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>{'Adaptation': [{'mal_id': 703, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Geneon Entertainment USA</td>
      <td>Victor Entertainment</td>
      <td>Madhouse</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Witch Hunter Robin</td>
      <td>Witches are individuals with special powers li...</td>
      <td>-</td>
      <td>Jul 2, 2002 to Dec 24, 2002</td>
      <td>0</td>
      <td>25 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>512</td>
      <td>...</td>
      <td>7.27</td>
      <td>37135</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Summer 2002</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>TV Tokyo, Bandai Visual, Dentsu, Victor Entert...</td>
      <td>Sunrise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Bouken Ou Beet</td>
      <td>It is the dark century and the people are suff...</td>
      <td>-</td>
      <td>Sep 30, 2004 to Sep 29, 2005</td>
      <td>0</td>
      <td>23 min per ep</td>
      <td>52</td>
      <td>TV</td>
      <td>10</td>
      <td>...</td>
      <td>6.97</td>
      <td>5463</td>
      <td>PG - Children</td>
      <td>Fall 2004</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>{'Adaptation': [{'mal_id': 1348, 'type': 'mang...</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>TV Tokyo, Dentsu</td>
      <td>Toei Animation</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df.shape
```




    (18162, 23)




```python
df.isnull().sum()
```




    mal_id         0
    title          0
    synopsis       0
    background     0
    aired          0
    airing         0
    duration       0
    episodes       0
    type           0
    favorites      0
    members        0
    rank           0
    popularity     0
    score          0
    scored_by      0
    rating         0
    premiered      0
    genres        67
    related        0
    status         0
    licensors      0
    producers      0
    studios        0
    dtype: int64




```python
#Handling null values
df[df['genres'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mal_id</th>
      <th>title</th>
      <th>synopsis</th>
      <th>background</th>
      <th>aired</th>
      <th>airing</th>
      <th>duration</th>
      <th>episodes</th>
      <th>type</th>
      <th>favorites</th>
      <th>...</th>
      <th>score</th>
      <th>scored_by</th>
      <th>rating</th>
      <th>premiered</th>
      <th>genres</th>
      <th>related</th>
      <th>status</th>
      <th>licensors</th>
      <th>producers</th>
      <th>studios</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9781</th>
      <td>28487</td>
      <td>Ikite Iru</td>
      <td>Tsuyoshi is 9 years old and had friends over t...</td>
      <td>-</td>
      <td>1996</td>
      <td>0</td>
      <td>15 min</td>
      <td>1</td>
      <td>OVA</td>
      <td>0</td>
      <td>...</td>
      <td>-1.00</td>
      <td>-1</td>
      <td>PG - Children</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>9831</th>
      <td>28653</td>
      <td>Maze</td>
      <td>Abstract stop motion animation by Tochka.</td>
      <td>-</td>
      <td>2012</td>
      <td>0</td>
      <td>2 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>1</td>
      <td>...</td>
      <td>5.60</td>
      <td>109</td>
      <td>G - All Ages</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>9832</th>
      <td>28655</td>
      <td>PiKA PiKA</td>
      <td>Abstract short film, the first "lightning dood...</td>
      <td>-</td>
      <td>2006</td>
      <td>0</td>
      <td>3 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>2</td>
      <td>...</td>
      <td>5.08</td>
      <td>268</td>
      <td>G - All Ages</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>10083</th>
      <td>29655</td>
      <td>Chanda Gou</td>
      <td>Independent animation by Yanagihara Ryouhei, m...</td>
      <td>-</td>
      <td>1964</td>
      <td>0</td>
      <td>7 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>0</td>
      <td>...</td>
      <td>-1.00</td>
      <td>-1</td>
      <td>G - All Ages</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>10138</th>
      <td>29765</td>
      <td>Metropolis (2009)</td>
      <td>Mirai Mizue's first time experimenting with ge...</td>
      <td>-</td>
      <td>2009</td>
      <td>0</td>
      <td>4 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>1</td>
      <td>...</td>
      <td>5.87</td>
      <td>223</td>
      <td>G - All Ages</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>Mirai Film</td>
      <td>-</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17448</th>
      <td>44041</td>
      <td>SD Gundam World Heroes</td>
      <td>The balance of the worlds is maintained by her...</td>
      <td>-</td>
      <td>Apr 8, 2021 to ?</td>
      <td>1</td>
      <td>24 min</td>
      <td>-1</td>
      <td>ONA</td>
      <td>0</td>
      <td>...</td>
      <td>5.68</td>
      <td>123</td>
      <td>PG - Children</td>
      <td>-</td>
      <td>NaN</td>
      <td>{'Other': [{'mal_id': 38715, 'type': 'anime', ...</td>
      <td>Currently Airing</td>
      <td>-</td>
      <td>-</td>
      <td>Sunrise</td>
    </tr>
    <tr>
      <th>17817</th>
      <td>47426</td>
      <td>23-ji no Saga Meshi Anime</td>
      <td>A series of short animations to promote the fo...</td>
      <td>-</td>
      <td>Feb 15, 2021 to Feb 25, 2021</td>
      <td>0</td>
      <td>19 sec per ep</td>
      <td>10</td>
      <td>ONA</td>
      <td>0</td>
      <td>...</td>
      <td>-1.00</td>
      <td>-1</td>
      <td>G - All Ages</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>17879</th>
      <td>48171</td>
      <td>Summer Ghost</td>
      <td>-</td>
      <td>-</td>
      <td>2021</td>
      <td>0</td>
      <td>Unknown</td>
      <td>1</td>
      <td>Movie</td>
      <td>2</td>
      <td>...</td>
      <td>-1.00</td>
      <td>-1</td>
      <td>None</td>
      <td>-</td>
      <td>NaN</td>
      <td>{'Other': [{'mal_id': 48177, 'type': 'anime', ...</td>
      <td>Not yet aired</td>
      <td>-</td>
      <td>-</td>
      <td>Flat Studio</td>
    </tr>
    <tr>
      <th>18046</th>
      <td>48644</td>
      <td>Gyakuten Sekai no Denchi Shoujo</td>
      <td>-</td>
      <td>-</td>
      <td>Not available</td>
      <td>0</td>
      <td>Unknown</td>
      <td>-1</td>
      <td>TV</td>
      <td>1</td>
      <td>...</td>
      <td>-1.00</td>
      <td>-1</td>
      <td>None</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Not yet aired</td>
      <td>-</td>
      <td>Egg Firm</td>
      <td>Lerche</td>
    </tr>
    <tr>
      <th>18068</th>
      <td>48708</td>
      <td>Suisou no Tora</td>
      <td>Something/someone is there. This fact seeks a ...</td>
      <td>-</td>
      <td>Not available</td>
      <td>0</td>
      <td>Unknown</td>
      <td>1</td>
      <td>Movie</td>
      <td>0</td>
      <td>...</td>
      <td>-1.00</td>
      <td>-1</td>
      <td>None</td>
      <td>-</td>
      <td>NaN</td>
      <td>{}</td>
      <td>Not yet aired</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 23 columns</p>
</div>




```python
#Fill null values with 'Unknown'
df['genres'].fillna('Unknown', inplace=True)
df.isnull().any()
```




    mal_id        False
    title         False
    synopsis      False
    background    False
    aired         False
    airing        False
    duration      False
    episodes      False
    type          False
    favorites     False
    members       False
    rank          False
    popularity    False
    score         False
    scored_by     False
    rating        False
    premiered     False
    genres        False
    related       False
    status        False
    licensors     False
    producers     False
    studios       False
    dtype: bool




```python
#Building the Recommendation engine
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mal_id</th>
      <th>title</th>
      <th>synopsis</th>
      <th>background</th>
      <th>aired</th>
      <th>airing</th>
      <th>duration</th>
      <th>episodes</th>
      <th>type</th>
      <th>favorites</th>
      <th>...</th>
      <th>score</th>
      <th>scored_by</th>
      <th>rating</th>
      <th>premiered</th>
      <th>genres</th>
      <th>related</th>
      <th>status</th>
      <th>licensors</th>
      <th>producers</th>
      <th>studios</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>In the year 2071, humanity has colonized sever...</td>
      <td>-</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>59968</td>
      <td>...</td>
      <td>8.77</td>
      <td>661519</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>Spring 1998</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>{'Adaptation': [{'mal_id': 173, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>Bandai Visual</td>
      <td>Sunrise</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>Another day, another bounty—such is the life o...</td>
      <td>-</td>
      <td>Sep 1, 2001</td>
      <td>0</td>
      <td>1 hr 55 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>1063</td>
      <td>...</td>
      <td>8.39</td>
      <td>168515</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>-</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>{'Parent story': [{'mal_id': 1, 'type': 'anime...</td>
      <td>Finished Airing</td>
      <td>Sony Pictures Entertainment</td>
      <td>Sunrise, Bandai Visual</td>
      <td>Bones</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>Vash the Stampede is the man with a $$60,000,0...</td>
      <td>-</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>11882</td>
      <td>...</td>
      <td>8.23</td>
      <td>288760</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Spring 1998</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>{'Adaptation': [{'mal_id': 703, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Geneon Entertainment USA</td>
      <td>Victor Entertainment</td>
      <td>Madhouse</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Witch Hunter Robin</td>
      <td>Witches are individuals with special powers li...</td>
      <td>-</td>
      <td>Jul 2, 2002 to Dec 24, 2002</td>
      <td>0</td>
      <td>25 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>512</td>
      <td>...</td>
      <td>7.27</td>
      <td>37135</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Summer 2002</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>TV Tokyo, Bandai Visual, Dentsu, Victor Entert...</td>
      <td>Sunrise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Bouken Ou Beet</td>
      <td>It is the dark century and the people are suff...</td>
      <td>-</td>
      <td>Sep 30, 2004 to Sep 29, 2005</td>
      <td>0</td>
      <td>23 min per ep</td>
      <td>52</td>
      <td>TV</td>
      <td>10</td>
      <td>...</td>
      <td>6.97</td>
      <td>5463</td>
      <td>PG - Children</td>
      <td>Fall 2004</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>{'Adaptation': [{'mal_id': 1348, 'type': 'mang...</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>TV Tokyo, Dentsu</td>
      <td>Toei Animation</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
#Calculate weighted scores to query similar type of animes
m = df.members.quantile(0.75)
C = df.score.mean()
print(m, C)
```

    13387.75 4.301492677017959



```python
def weighted_score(df, m, C):
    term = df['members'] / (m + df['members'])
    return df['score'] * term + (1-term) * C
```


```python
df['community_score'] = df.apply(weighted_score, axis=1, args=(m,C))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mal_id</th>
      <th>title</th>
      <th>synopsis</th>
      <th>background</th>
      <th>aired</th>
      <th>airing</th>
      <th>duration</th>
      <th>episodes</th>
      <th>type</th>
      <th>favorites</th>
      <th>...</th>
      <th>scored_by</th>
      <th>rating</th>
      <th>premiered</th>
      <th>genres</th>
      <th>related</th>
      <th>status</th>
      <th>licensors</th>
      <th>producers</th>
      <th>studios</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Cowboy Bebop</td>
      <td>In the year 2071, humanity has colonized sever...</td>
      <td>-</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>59968</td>
      <td>...</td>
      <td>661519</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>Spring 1998</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>{'Adaptation': [{'mal_id': 173, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>Bandai Visual</td>
      <td>Sunrise</td>
      <td>8.726639</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>Another day, another bounty—such is the life o...</td>
      <td>-</td>
      <td>Sep 1, 2001</td>
      <td>0</td>
      <td>1 hr 55 min</td>
      <td>1</td>
      <td>Movie</td>
      <td>1063</td>
      <td>...</td>
      <td>168515</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>-</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>{'Parent story': [{'mal_id': 1, 'type': 'anime...</td>
      <td>Finished Airing</td>
      <td>Sony Pictures Entertainment</td>
      <td>Sunrise, Bandai Visual</td>
      <td>Bones</td>
      <td>8.210329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Trigun</td>
      <td>Vash the Stampede is the man with a $$60,000,0...</td>
      <td>-</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>11882</td>
      <td>...</td>
      <td>288760</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Spring 1998</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>{'Adaptation': [{'mal_id': 703, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Geneon Entertainment USA</td>
      <td>Victor Entertainment</td>
      <td>Madhouse</td>
      <td>8.142999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Witch Hunter Robin</td>
      <td>Witches are individuals with special powers li...</td>
      <td>-</td>
      <td>Jul 2, 2002 to Dec 24, 2002</td>
      <td>0</td>
      <td>25 min per ep</td>
      <td>26</td>
      <td>TV</td>
      <td>512</td>
      <td>...</td>
      <td>37135</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Summer 2002</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>TV Tokyo, Bandai Visual, Dentsu, Victor Entert...</td>
      <td>Sunrise</td>
      <td>6.912250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Bouken Ou Beet</td>
      <td>It is the dark century and the people are suff...</td>
      <td>-</td>
      <td>Sep 30, 2004 to Sep 29, 2005</td>
      <td>0</td>
      <td>23 min per ep</td>
      <td>52</td>
      <td>TV</td>
      <td>10</td>
      <td>...</td>
      <td>5463</td>
      <td>PG - Children</td>
      <td>Fall 2004</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>{'Adaptation': [{'mal_id': 1348, 'type': 'mang...</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>TV Tokyo, Dentsu</td>
      <td>Toei Animation</td>
      <td>5.645062</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
#Dropping unused columns 
df.drop(['mal_id', 'score', 'members', 'episodes'], axis=1, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>synopsis</th>
      <th>background</th>
      <th>aired</th>
      <th>airing</th>
      <th>duration</th>
      <th>type</th>
      <th>favorites</th>
      <th>rank</th>
      <th>popularity</th>
      <th>scored_by</th>
      <th>rating</th>
      <th>premiered</th>
      <th>genres</th>
      <th>related</th>
      <th>status</th>
      <th>licensors</th>
      <th>producers</th>
      <th>studios</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cowboy Bebop</td>
      <td>In the year 2071, humanity has colonized sever...</td>
      <td>-</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>TV</td>
      <td>59968</td>
      <td>32.0</td>
      <td>44</td>
      <td>661519</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>Spring 1998</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>
      <td>{'Adaptation': [{'mal_id': 173, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>Bandai Visual</td>
      <td>Sunrise</td>
      <td>8.726639</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>Another day, another bounty—such is the life o...</td>
      <td>-</td>
      <td>Sep 1, 2001</td>
      <td>0</td>
      <td>1 hr 55 min</td>
      <td>Movie</td>
      <td>1063</td>
      <td>163.0</td>
      <td>542</td>
      <td>168515</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>-</td>
      <td>Action, Drama, Mystery, Sci-Fi, Space</td>
      <td>{'Parent story': [{'mal_id': 1, 'type': 'anime...</td>
      <td>Finished Airing</td>
      <td>Sony Pictures Entertainment</td>
      <td>Sunrise, Bandai Visual</td>
      <td>Bones</td>
      <td>8.210329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trigun</td>
      <td>Vash the Stampede is the man with a $$60,000,0...</td>
      <td>-</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>TV</td>
      <td>11882</td>
      <td>285.0</td>
      <td>213</td>
      <td>288760</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Spring 1998</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>{'Adaptation': [{'mal_id': 703, 'type': 'manga...</td>
      <td>Finished Airing</td>
      <td>Funimation, Geneon Entertainment USA</td>
      <td>Victor Entertainment</td>
      <td>Madhouse</td>
      <td>8.142999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Witch Hunter Robin</td>
      <td>Witches are individuals with special powers li...</td>
      <td>-</td>
      <td>Jul 2, 2002 to Dec 24, 2002</td>
      <td>0</td>
      <td>25 min per ep</td>
      <td>TV</td>
      <td>512</td>
      <td>2508.0</td>
      <td>1535</td>
      <td>37135</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>Summer 2002</td>
      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>
      <td>{}</td>
      <td>Finished Airing</td>
      <td>Funimation, Bandai Entertainment</td>
      <td>TV Tokyo, Bandai Visual, Dentsu, Victor Entert...</td>
      <td>Sunrise</td>
      <td>6.912250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bouken Ou Beet</td>
      <td>It is the dark century and the people are suff...</td>
      <td>-</td>
      <td>Sep 30, 2004 to Sep 29, 2005</td>
      <td>0</td>
      <td>23 min per ep</td>
      <td>TV</td>
      <td>10</td>
      <td>3769.0</td>
      <td>4514</td>
      <td>5463</td>
      <td>PG - Children</td>
      <td>Fall 2004</td>
      <td>Adventure, Fantasy, Shounen, Supernatural</td>
      <td>{'Adaptation': [{'mal_id': 1348, 'type': 'mang...</td>
      <td>Finished Airing</td>
      <td>-</td>
      <td>TV Tokyo, Dentsu</td>
      <td>Toei Animation</td>
      <td>5.645062</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Concat Genre and Type as equal for better recommendation
df = pd.concat([df, df['type'].str.get_dummies(), df['genres'].str.get_dummies(sep=',')], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>synopsis</th>
      <th>background</th>
      <th>aired</th>
      <th>airing</th>
      <th>duration</th>
      <th>type</th>
      <th>favorites</th>
      <th>rank</th>
      <th>popularity</th>
      <th>...</th>
      <th>Thriller</th>
      <th>Unknown</th>
      <th>Vampire</th>
      <th>Yaoi</th>
      <th>['Action'</th>
      <th>['Comedy']</th>
      <th>['Dementia'</th>
      <th>['Music'</th>
      <th>['Sci-Fi'</th>
      <th>['Slice of Life']</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cowboy Bebop</td>
      <td>In the year 2071, humanity has colonized sever...</td>
      <td>-</td>
      <td>Apr 3, 1998 to Apr 24, 1999</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>TV</td>
      <td>59968</td>
      <td>32.0</td>
      <td>44</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cowboy Bebop: Tengoku no Tobira</td>
      <td>Another day, another bounty—such is the life o...</td>
      <td>-</td>
      <td>Sep 1, 2001</td>
      <td>0</td>
      <td>1 hr 55 min</td>
      <td>Movie</td>
      <td>1063</td>
      <td>163.0</td>
      <td>542</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trigun</td>
      <td>Vash the Stampede is the man with a $$60,000,0...</td>
      <td>-</td>
      <td>Apr 1, 1998 to Sep 30, 1998</td>
      <td>0</td>
      <td>24 min per ep</td>
      <td>TV</td>
      <td>11882</td>
      <td>285.0</td>
      <td>213</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Witch Hunter Robin</td>
      <td>Witches are individuals with special powers li...</td>
      <td>-</td>
      <td>Jul 2, 2002 to Dec 24, 2002</td>
      <td>0</td>
      <td>25 min per ep</td>
      <td>TV</td>
      <td>512</td>
      <td>2508.0</td>
      <td>1535</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bouken Ou Beet</td>
      <td>It is the dark century and the people are suff...</td>
      <td>-</td>
      <td>Sep 30, 2004 to Sep 29, 2005</td>
      <td>0</td>
      <td>23 min per ep</td>
      <td>TV</td>
      <td>10</td>
      <td>3769.0</td>
      <td>4514</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 128 columns</p>
</div>




```python
anime_features = df.loc[:, "Movie":].copy()
anime_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie</th>
      <th>Music</th>
      <th>ONA</th>
      <th>OVA</th>
      <th>Special</th>
      <th>TV</th>
      <th>Unknown</th>
      <th>'Adventure'</th>
      <th>'Demons']</th>
      <th>'Drama'</th>
      <th>...</th>
      <th>Thriller</th>
      <th>Unknown</th>
      <th>Vampire</th>
      <th>Yaoi</th>
      <th>['Action'</th>
      <th>['Comedy']</th>
      <th>['Dementia'</th>
      <th>['Music'</th>
      <th>['Sci-Fi'</th>
      <th>['Slice of Life']</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 108 columns</p>
</div>




```python
#Calculate similarity Matrix
cosine_sim = cosine_similarity(anime_features.values, anime_features.values)
```


```python
cosine_sim
```




    array([[1.        , 0.6172134 , 0.85714286, ..., 0.18898224, 0.        ,
            0.        ],
           [0.6172134 , 1.        , 0.46291005, ..., 0.        , 0.        ,
            0.20412415],
           [0.85714286, 0.46291005, 1.        , ..., 0.18898224, 0.        ,
            0.        ],
           ...,
           [0.18898224, 0.        , 0.18898224, ..., 1.        , 0.40824829,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.40824829, 1.        ,
            0.        ],
           [0.        , 0.20412415, 0.        , ..., 0.        , 0.        ,
            1.        ]])




```python
cosine_sim.shape
```




    (18162, 18162)




```python
anime_index = pd.Series(df.index, index=df.title).drop_duplicates()
```


```python
def get_recommendation(anime_name, similarity=cosine_sim):
    idx = anime_index[anime_name]
    
    # Get the pairwsie similarity scores of all anime with that anime
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort anime based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get scores of the 10 most similar anime
    sim_scores = sim_scores[0:11]

    # Get anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar anime
    result = df[['title', 'genres', 'rank','community_score']].iloc[anime_indices].drop(idx)
    return result
```


```python
#Search for Recommendations
get_recommendation("Cowboy Bebop")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rank</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1114</th>
      <td>Seihou Tenshi Angel Links</td>
      <td>Action, Adventure, Space, Comedy, Romance, Dra...</td>
      <td>8518.0</td>
      <td>4.905682</td>
    </tr>
    <tr>
      <th>376</th>
      <td>Seihou Bukyou Outlaw Star</td>
      <td>Action, Sci-Fi, Adventure, Space, Comedy</td>
      <td>731.0</td>
      <td>7.527605</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>Ginga Tetsudou Monogatari</td>
      <td>Action, Adventure, Drama, Sci-Fi, Space</td>
      <td>3085.0</td>
      <td>5.521256</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>Waga Seishun no Arcadia: Mugen Kidou SSX</td>
      <td>Action, Adventure, Drama, Sci-Fi, Space</td>
      <td>1916.0</td>
      <td>5.201664</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>Ginga Tetsudou Monogatari: Eien e no Bunkiten</td>
      <td>Action, Adventure, Space, Drama, Sci-Fi</td>
      <td>4415.0</td>
      <td>4.807607</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trigun</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>
      <td>285.0</td>
      <td>8.142999</td>
    </tr>
    <tr>
      <th>186</th>
      <td>R.O.D: The TV</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Super Power...</td>
      <td>1434.0</td>
      <td>6.978488</td>
    </tr>
    <tr>
      <th>901</th>
      <td>Uchuu Kaizoku Captain Herlock</td>
      <td>Action, Sci-Fi, Adventure, Space, Drama, Seinen</td>
      <td>1032.0</td>
      <td>6.648883</td>
    </tr>
    <tr>
      <th>923</th>
      <td>Generator Gawl</td>
      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Shounen</td>
      <td>4082.0</td>
      <td>5.304560</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>Urusei Yatsura</td>
      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Romance</td>
      <td>1081.0</td>
      <td>7.072141</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_recommendation("Dragon Ball Z")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rank</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4422</th>
      <td>Dragon Ball Kai</td>
      <td>Action, Adventure, Comedy, Fantasy, Martial Ar...</td>
      <td>999.0</td>
      <td>7.560977</td>
    </tr>
    <tr>
      <th>8767</th>
      <td>Dragon Ball Kai (2014)</td>
      <td>Action, Adventure, Comedy, Super Power, Martia...</td>
      <td>1111.0</td>
      <td>7.374705</td>
    </tr>
    <tr>
      <th>10616</th>
      <td>Dragon Ball Super</td>
      <td>Action, Adventure, Comedy, Super Power, Martia...</td>
      <td>1933.0</td>
      <td>7.332894</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Naruto</td>
      <td>Action, Adventure, Comedy, Super Power, Martia...</td>
      <td>637.0</td>
      <td>7.906696</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>Naruto: Shippuuden</td>
      <td>Action, Adventure, Comedy, Super Power, Martia...</td>
      <td>326.0</td>
      <td>8.160268</td>
    </tr>
    <tr>
      <th>13350</th>
      <td>Shaolin Chuanqi</td>
      <td>Action, Adventure, Comedy, Fantasy, Martial Ar...</td>
      <td>12497.0</td>
      <td>4.266091</td>
    </tr>
    <tr>
      <th>11</th>
      <td>One Piece</td>
      <td>Action, Adventure, Comedy, Super Power, Drama,...</td>
      <td>83.0</td>
      <td>8.512651</td>
    </tr>
    <tr>
      <th>815</th>
      <td>Dragon Ball Z Movie 11: Super Senshi Gekiha!! ...</td>
      <td>Action, Adventure, Comedy, Fantasy, Martial Ar...</td>
      <td>8862.0</td>
      <td>5.672063</td>
    </tr>
    <tr>
      <th>888</th>
      <td>Dragon Ball GT: Gokuu Gaiden! Yuuki no Akashi ...</td>
      <td>Action, Adventure, Comedy, Super Power, Martia...</td>
      <td>5799.0</td>
      <td>6.128450</td>
    </tr>
    <tr>
      <th>4710</th>
      <td>Dragon Ball Z: Atsumare! Gokuu World</td>
      <td>Action, Adventure, Comedy, Super Power, Martia...</td>
      <td>6099.0</td>
      <td>5.553625</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_recommendation("Pokemon")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rank</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>517</th>
      <td>Digimon Adventure</td>
      <td>Action, Adventure, Comedy, Fantasy, Kids</td>
      <td>875.0</td>
      <td>7.648739</td>
    </tr>
    <tr>
      <th>1416</th>
      <td>Pokemon Advanced Generation</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>3299.0</td>
      <td>6.861242</td>
    </tr>
    <tr>
      <th>1417</th>
      <td>Pokemon Diamond &amp; Pearl</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>2771.0</td>
      <td>6.896972</td>
    </tr>
    <tr>
      <th>3944</th>
      <td>Croket!</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>4094.0</td>
      <td>4.466920</td>
    </tr>
    <tr>
      <th>5637</th>
      <td>Pokemon Best Wishes!</td>
      <td>Action, Adventure, Comedy, Fantasy, Kids</td>
      <td>6544.0</td>
      <td>6.083322</td>
    </tr>
    <tr>
      <th>6245</th>
      <td>Duel Masters Victory</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>9071.0</td>
      <td>4.420559</td>
    </tr>
    <tr>
      <th>6248</th>
      <td>Duel Masters Cross</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>6880.0</td>
      <td>4.560186</td>
    </tr>
    <tr>
      <th>7086</th>
      <td>Pokemon Best Wishes! Season 2</td>
      <td>Action, Adventure, Comedy, Fantasy, Kids</td>
      <td>6545.0</td>
      <td>5.953492</td>
    </tr>
    <tr>
      <th>7585</th>
      <td>Pokemon Best Wishes! Season 2: Episode N</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>5117.0</td>
      <td>5.997336</td>
    </tr>
    <tr>
      <th>7758</th>
      <td>Pokemon Best Wishes! Season 2: Decolora Adventure</td>
      <td>Action, Adventure, Comedy, Kids, Fantasy</td>
      <td>7110.0</td>
      <td>5.691909</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_recommendation("Akira")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rank</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6683</th>
      <td>Berserk: Ougon Jidai-hen II - Doldrey Kouryaku</td>
      <td>Action, Adventure, Demons, Drama, Fantasy, Hor...</td>
      <td>686.0</td>
      <td>7.623915</td>
    </tr>
    <tr>
      <th>6684</th>
      <td>Berserk: Ougon Jidai-hen III - Kourin</td>
      <td>Action, Adventure, Demons, Drama, Fantasy, Hor...</td>
      <td>309.0</td>
      <td>7.913440</td>
    </tr>
    <tr>
      <th>6111</th>
      <td>Berserk: Ougon Jidai-hen I - Haou no Tamago</td>
      <td>Action, Military, Adventure, Demons, Supernatu...</td>
      <td>949.0</td>
      <td>7.496090</td>
    </tr>
    <tr>
      <th>8750</th>
      <td>Appleseed Alpha</td>
      <td>Action, Adventure, Police, Mecha, Military, Sc...</td>
      <td>3616.0</td>
      <td>5.783379</td>
    </tr>
    <tr>
      <th>672</th>
      <td>Vampire Hunter D</td>
      <td>Action, Sci-Fi, Horror, Supernatural, Vampire</td>
      <td>3388.0</td>
      <td>6.609581</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>Golgo 13</td>
      <td>Action, Military, Adventure, Drama, Seinen</td>
      <td>4580.0</td>
      <td>5.563073</td>
    </tr>
    <tr>
      <th>10684</th>
      <td>Ajin Part 1: Shoudou</td>
      <td>Action, Horror, Mystery, Seinen, Supernatural</td>
      <td>1602.0</td>
      <td>6.693244</td>
    </tr>
    <tr>
      <th>10685</th>
      <td>Ajin Part 2: Shoutotsu</td>
      <td>Action, Horror, Mystery, Seinen, Supernatural</td>
      <td>2536.0</td>
      <td>6.167124</td>
    </tr>
    <tr>
      <th>10686</th>
      <td>Ajin Part 3: Shougeki</td>
      <td>Action, Horror, Mystery, Seinen, Supernatural</td>
      <td>2440.0</td>
      <td>6.130046</td>
    </tr>
    <tr>
      <th>15919</th>
      <td>Akira (Shin Anime)</td>
      <td>Action, Military, Sci-Fi, Supernatural, Seinen</td>
      <td>-1.0</td>
      <td>1.673890</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_recommendation("Kimi no Na wa.")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rank</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>513</th>
      <td>Wind: A Breath of Heart OVA</td>
      <td>Romance, Supernatural, School, Drama</td>
      <td>7354.0</td>
      <td>4.619994</td>
    </tr>
    <tr>
      <th>10871</th>
      <td>Zutto Mae kara Suki deshita.: Kokuhaku Jikkou ...</td>
      <td>Romance, School</td>
      <td>2535.0</td>
      <td>7.031393</td>
    </tr>
    <tr>
      <th>12721</th>
      <td>Hakubo</td>
      <td>Romance, School</td>
      <td>6527.0</td>
      <td>5.514123</td>
    </tr>
    <tr>
      <th>16532</th>
      <td>Nakitai Watashi wa Neko wo Kaburu</td>
      <td>Comedy, Supernatural, Drama, Romance, School</td>
      <td>2125.0</td>
      <td>7.179328</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Sen to Chihiro no Kamikakushi</td>
      <td>Adventure, Supernatural, Drama</td>
      <td>25.0</td>
      <td>8.766568</td>
    </tr>
    <tr>
      <th>691</th>
      <td>School Days ONA</td>
      <td>Romance, School, Drama</td>
      <td>8652.0</td>
      <td>5.446190</td>
    </tr>
    <tr>
      <th>940</th>
      <td>Mizuiro (2003)</td>
      <td>Romance, Supernatural, Drama</td>
      <td>6535.0</td>
      <td>4.720896</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>To Heart 2 Special</td>
      <td>Romance, School, Drama</td>
      <td>6640.0</td>
      <td>4.725616</td>
    </tr>
    <tr>
      <th>5232</th>
      <td>Gotou ni Naritai.</td>
      <td>Slice of Life, School, Drama</td>
      <td>3365.0</td>
      <td>4.701045</td>
    </tr>
    <tr>
      <th>7098</th>
      <td>Aitsu to Lullaby: Suiyobi no Cinderella</td>
      <td>Romance, Drama, Shounen</td>
      <td>8718.0</td>
      <td>4.362697</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_recommendation("Death Note")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
      <th>rank</th>
      <th>community_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1716</th>
      <td>Higurashi no Naku Koro ni Kai</td>
      <td>Mystery, Psychological, Supernatural, Thriller</td>
      <td>302.0</td>
      <td>8.085291</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Mousou Dairinin</td>
      <td>Mystery, Dementia, Police, Psychological, Supe...</td>
      <td>1076.0</td>
      <td>7.540075</td>
    </tr>
    <tr>
      <th>2727</th>
      <td>Death Note: Rewrite</td>
      <td>Mystery, Police, Psychological, Supernatural, ...</td>
      <td>1035.0</td>
      <td>7.442780</td>
    </tr>
    <tr>
      <th>3864</th>
      <td>Mouryou no Hako</td>
      <td>Mystery, Psychological, Supernatural, Thriller...</td>
      <td>2865.0</td>
      <td>6.611852</td>
    </tr>
    <tr>
      <th>16418</th>
      <td>Yuukoku no Moriarty</td>
      <td>Mystery, Historical, Psychological, Thriller, ...</td>
      <td>461.0</td>
      <td>7.846126</td>
    </tr>
    <tr>
      <th>17822</th>
      <td>Yakusoku no Neverland 2nd Season: Michishirube</td>
      <td>Mystery, Psychological, Supernatural, Thriller...</td>
      <td>11294.0</td>
      <td>4.530100</td>
    </tr>
    <tr>
      <th>2555</th>
      <td>Saint Luminous Jogakuin</td>
      <td>Mystery, Psychological, Supernatural</td>
      <td>8183.0</td>
      <td>4.583642</td>
    </tr>
    <tr>
      <th>3165</th>
      <td>Yakushiji Ryouko no Kaiki Jikenbo</td>
      <td>Mystery, Police, Supernatural</td>
      <td>3309.0</td>
      <td>5.827840</td>
    </tr>
    <tr>
      <th>3234</th>
      <td>Jigoku Shoujo Mitsuganae</td>
      <td>Mystery, Psychological, Supernatural</td>
      <td>1196.0</td>
      <td>7.212237</td>
    </tr>
    <tr>
      <th>8881</th>
      <td>Zankyou no Terror</td>
      <td>Mystery, Psychological, Thriller</td>
      <td>395.0</td>
      <td>8.063463</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
