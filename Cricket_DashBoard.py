import streamlit as st
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import time
from urllib.parse import urljoin

# Window View
st.set_page_config(layout="wide")

# PLAYER IMAGE
def display_player_image(player_name):
    # Image URL of the webpage
    image_page_url = "https://en.wikipedia.org/wiki/File:" + player_name + ".jpg"
    # Send an HTTP GET request to the webpage
    image_response = requests.get(image_page_url)
    # Check if the request was successful (status code 200)
    if image_response.status_code == 200:
        # Create a BeautifulSoup object from the webpage content
        soup = BeautifulSoup(image_response.content, "html.parser")
        # Find the <a> tag containing the image URL
        a_tag = soup.find("a", class_="internal")
        # Check if the <a> tag was found
        if a_tag:
            # Get the href attribute of the <a> tag
            image_url = a_tag["href"]
            # Join the URL of the webpage and the image URL to get the absolute URL
            absolute_url = urljoin(image_page_url, image_url)
            # Check if the request was successful (status code 200)
            if image_response.status_code == 200:
                return absolute_url
    return None


# Web-Scrapping
def Player():
    # Make a request to the URL
    url = 'https://en.wikipedia.org/wiki/' + player_name
    response = requests.get(url)
    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


# GENERAL INFORMATION OF PLAYER
def General_Info():
    # Find the infobox table
    infobox = Player().find('table', class_='infobox')
    # Initialize the data lists
    labels = []
    values = []
    # GENERAL INFORMATION
    for row in infobox.find_all('tr'):
        # Find the header cell and extract the label
        header_cell = row.find('th', scope='row')
        if header_cell:
            label = header_cell.text.strip()
            labels.append(label)

            # Find the data cell and extract the value
            data_cell = row.find('td')
            if data_cell:
                value = data_cell.text.strip()
                values.append(value)
    # Create a DataFrame from the data lists
    General_information = pd.DataFrame({'Label': labels, 'Information': values})
    return (General_information)


# CAREER STATISTICS
def Career():
    career_stats_table = Player().find('table', style='width:100%; margin:-1px; white-space:nowrap;')
    # Initialize the career statistics list
    career_stats = []
    # Find the competition names from the first row
    competition_names = [th.text.strip() for th in career_stats_table.find('tr').find_all('th')]
    # Iterate over the rows in the career statistics table
    for row in career_stats_table.find_all('tr')[1:]:
        # Extract the data cells in each row
        cells = row.find_all('td')
        name_cell = row.find('th')
        # Check if it's a valid row with data cells
        if len(cells) >= 4 and name_cell:
            name = name_cell.text.strip()
            test = cells[0].text.strip()
            odi = cells[1].text.strip()
            t20i = cells[2].text.strip()
            # Handle missing data cell for T20
            try:
                t20 = cells[3].text.strip()
            except IndexError:
                t20 = ''
            # Add the career statistics to the list
            career_stats.append([name, test, odi, t20i, t20])
        # Create a DataFrame for the career statistics
        Career_Stats = pd.DataFrame(career_stats, columns=['Name'] + competition_names[1:])
    return Career_Stats


def International_Awards():
    # AWARDS
    awards_table = Player().find('table',
                                 style='width:100%; background-color:#f9f9f9; color:#000000; font-weight:normal;')
    awards = []
    if awards_table is not None:
        # Iterate over the rows in the awards table
        for row in awards_table.find_all('tr')[1:]:
            # Extract the data cells in each row
            cells = row.find_all('td')

            # Check if it's a valid row with data cells
            if len(cells) >= 3:
                award_position = cells[0].text.strip()
                award_name = cells[1].text.strip()
                competition = row.find('a')['title']

                # Check if an image is present for the position column
                position_img = cells[0].find('img')
                if position_img is not None:
                    image_alt = position_img['alt']
                    # Extract the image name from the alt attribute
                    image_name = image_alt.split('/')[-1]
                    award_position = image_name

                # Add the awards to the list
                awards.append([competition, award_name, award_position])

        # Create a DataFrame for the awards
        Awards_df = pd.DataFrame(awards, columns=['Competition', 'Award Name', 'Position'])
        return Awards_df
    else:
        return None


def IPL_Awards():
    # IPL
    awards_ipl = Player().find('table', class_='infobox', style='width:22em;')
    ipl = []
    if awards_ipl is not None:
        # Iterate over the rows in the awards table
        for row in awards_ipl.find_all('tr')[1:]:
            # Extract the data cells in each row
            cells = row.find_all('td')

            # Check if it's a valid row with data cells
            if len(cells) >= 3:
                award_position = cells[0].text.strip()
                award_name = cells[1].text.strip()

                # Check if the competition name link exists
                competition_link = row.find('a')
                if competition_link is not None:
                    competition = competition_link['title']
                else:
                    competition = ""

                # Add the awards to the list
                ipl.append([competition, award_name, award_position])

        # Create a DataFrame for the awards
        IPL_Awards = pd.DataFrame(ipl, columns=['Competition', 'Award Name', 'Position'])
        return IPL_Awards
    else:
        return None


# Born
def Born():
    born = General_Info()[General_Info()['Label'].str.contains('Born', case=False, na=False)]['Information'].values[0]
    # Extract the birthday date using regex pattern matching
    birthday = re.search(r'(\d{1,2}\s\w+\s\d{4})', born).group(1)
    # Extract the birthplace using regex pattern matching
    birthplace = re.search(r"(\d+\s\w+\s\d+)\s.*?(\w+,\s[\w\s]+)", born).group(2)
    return (birthday, birthplace)


# Height

def Height():
    # Check if the label contains the keyword "Height"
    height_label = General_Info()[General_Info()['Label'].str.contains('Height', case=False, na=False)]
    if not height_label.empty:
        # Extract the height using regex pattern matching
        height_match = re.search(r"(\d+)\sft\s(\d+)\sin", height_label['Information'].values[0])
        if height_match:
            feet = height_match.group(1)
            inches = height_match.group(2)
            height = f"{feet} ft {inches} in"
            return height
        else:
            return '---'
    else:
        return '---'

# Debut
def Debut():
    debut_values = General_Info()[General_Info()['Label'].str.contains('debut', case=False)]
    df_debut = pd.DataFrame({'Debut': debut_values['Label'], 'Information': debut_values['Information']})
    df_debut = df_debut.reset_index(drop=True)
    # Extract country from "Information" column
    df_debut['Against'] = df_debut['Information'].str.extract(r'v\s(.+)$')
    # Extract date with month and year from "Information" column
    df_debut['Day'] = df_debut['Information'].apply(lambda x: re.search(r'(\d+\s\w+\s\d+)', x).group(1))
    # Remove the "Information"
    Debut = df_debut.drop(['Information'], axis=1)
    return Debut


# Last
def Last():
    Last_values = General_Info()[General_Info()['Label'].str.contains('last', case=False)]
    df_last = pd.DataFrame({'Last': Last_values['Label'], 'Information': Last_values['Information']})
    df_last = df_last.reset_index(drop=True)
    # Extract country from "Information" column
    df_last['Against'] = df_last['Information'].str.extract(r'v\s(.+)$')
    # Extract date with month and year from "Information" column
    df_last['Day'] = df_last['Information'].apply(lambda x: re.search(r'(\d+\s\w+\s\d+)', x).group(1))
    # Remove the "Information"
    Last = df_last.drop(['Information'], axis=1)
    return Last


# GRAPHS

# Total matches
def Pie_Match_Played():
    career_df = Career()
    if 'T20' in career_df:
        values = [career_df['Test'][0], career_df['ODI'][0], career_df['T20'][0], career_df['T20I'][0]]
        labels = ['Test', 'ODI', 'T20', 'T20I']
    elif 'LA' in career_df:
        values = [career_df['Test'][0], career_df['ODI'][0], career_df['FC'][0], career_df['LA'][0]]
        labels = ['Test', 'ODI', 'FC', 'LA']
    elif 'IPL' in career_df:
        values = [career_df['Test'][0], career_df['ODI'][0], career_df['T20I'][0], career_df['IPL'][0]]
        labels = ['Test', 'ODI', 'FC', 'LA']
    else:
        values = [career_df['Test'][0], career_df['ODI'][0], career_df['FC'][0], career_df['T20I'][0]]
        labels = ['Test', 'ODI', 'FC', 'T20I']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    # Create pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors,
                                                                           line=dict(color='#000000', width=2)))])
    # Customize layout
    fig.update_layout(
        font=dict(family='Arial', size=24, color='#333333'),
        legend=dict(
            title='Legend',
            orientation='v',
            x=0.9, y=0.5,
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            font=dict(family='Arial', size=12),
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=50, r=50, t=80, b=50),
        height=500,  # Set the height of the graph (in pixels)
        width=900,  # Set the width of the graph (in pixels)
        title=dict(
            text='Total Matches Played',
            x=0.5,  # Set the x-position of the title to 0.5 (center)
            y=0.95,  # Set the y-position of the title
            xanchor='center',  # Anchor the x-position to the center
            yanchor='top'  # Anchor the y-position to the top
        )
    )
    # Customize pie chart
    fig.update_traces(
        textposition='inside',
        textinfo='percent',
        hoverinfo='label+percent+value',
    )
    return fig


# Total Runs Scored
def Pie_Runs_Scored():
    century = []
    half_century = []
    career_df = Career()
    if 'T20' in career_df:
        run_values = [career_df['Test'][1], career_df['ODI'][1], career_df['T20'][1], career_df['T20I'][1]]
        run_average = [career_df['Test'][2], career_df['ODI'][2], career_df['T20'][2], career_df['T20I'][2]]
        high_run = [career_df['Test'][4], career_df['ODI'][4], career_df['T20'][4], career_df['T20I'][4]]
        run_values = [int(size.replace(',', '')) for size in run_values]
        run_labels = ['Test', 'ODI', 'T20', 'T20I']
        labels = ['Test', 'ODI', 'T20I', 'T20']
        for format_name in labels:
            centuries_half_centuries = career_df.loc[career_df['Name'] == '100s/50s', format_name].values
            centuries_half_centuries = centuries_half_centuries[0].split('/')
            century.append({'format': format_name, 'centuries': centuries_half_centuries[0]})
            half_century.append({'format': format_name, 'half_centuries': centuries_half_centuries[1]})
            cent = [int(item['centuries']) for item in century]
            half_cent = [int(item['half_centuries']) for item in half_century]

    elif 'LA' in career_df:
        run_values = [career_df['Test'][1], career_df['ODI'][1], career_df['FC'][1], career_df['LA'][1]]
        run_average = [career_df['Test'][2], career_df['ODI'][2], career_df['FC'][2], career_df['LA'][2]]
        high_run = [career_df['Test'][4], career_df['ODI'][4], career_df['FC'][4], career_df['LA'][4]]
        run_values = [int(size.replace(',', '')) for size in run_values]
        run_labels = ['Test', 'ODI', 'FC', 'LA']
        labels = ['Test', 'ODI', 'FC', 'LA']
        for format_name in labels:
            centuries_half_centuries = career_df.loc[career_df['Name'] == '100s/50s', format_name].values
            centuries_half_centuries = centuries_half_centuries[0].split('/')
            century.append({'format': format_name, 'centuries': centuries_half_centuries[0]})
            half_century.append({'format': format_name, 'half_centuries': centuries_half_centuries[1]})
            cent = [int(item['centuries']) for item in century]
            half_cent = [int(item['half_centuries']) for item in half_century]
    elif 'IPL' in career_df:
        run_values = [career_df['Test'][1], career_df['ODI'][1], career_df['T20I'][1], career_df['IPL'][1]]
        run_average = [career_df['Test'][2], career_df['ODI'][2], career_df['T20I'][2], career_df['IPL'][2]]
        high_run = [career_df['Test'][4], career_df['ODI'][4], career_df['T20I'][4], career_df['IPL'][4]]
        run_values = [int(size.replace(',', '')) for size in run_values]
        run_labels = ['Test', 'ODI', 'T20I', 'IPL']
        labels = ['Test', 'ODI', 'T20I', 'IPL']
        for format_name in labels:
            centuries_half_centuries = career_df.loc[career_df['Name'] == '100s/50s', format_name].values
            centuries_half_centuries = centuries_half_centuries[0].split('/')
            century.append({'format': format_name, 'centuries': centuries_half_centuries[0]})
            half_century.append({'format': format_name, 'half_centuries': centuries_half_centuries[1]})
            cent = [int(item['centuries']) for item in century]
            half_cent = [int(item['half_centuries']) for item in half_century]
    else:
        run_values = [career_df['Test'][1], career_df['ODI'][1], career_df['FC'][1], career_df['T20I'][1]]
        run_average = [career_df['Test'][2], career_df['ODI'][2], career_df['FC'][2], career_df['T20I'][2]]
        high_run = [career_df['Test'][4], career_df['ODI'][4], career_df['FC'][4], career_df['T20I'][4]]
        run_values = [int(size.replace(',', '')) for size in run_values]
        run_labels = ['Test', 'ODI', 'FC', 'T20I']
        labels = ['Test', 'ODI', 'FC', 'T20I']
        for format_name in labels:
            centuries_half_centuries = career_df.loc[career_df['Name'] == '100s/50s', format_name].values
            centuries_half_centuries = centuries_half_centuries[0].split('/')
            century.append({'format': format_name, 'centuries': centuries_half_centuries[0]})
            half_century.append({'format': format_name, 'half_centuries': centuries_half_centuries[1]})
            cent = [int(item['centuries']) for item in century]
            half_cent = [int(item['half_centuries']) for item in half_century]

    return run_values, run_labels, run_average, high_run, cent, half_cent


# Bowling Analysed
def Pie_Balls():
    career_df = Career()
    not_played = 'Not Played'  # New value for the "-" entries
    special_char = '–'

    if len(career_df) > 7:
        if 'T20' in career_df:
            balls_bowled = [not_played if size == special_char else size.replace(',', '') if ',' in size else size for
                            size
                            in [career_df['Test'][5], career_df['ODI'][5], career_df['T20'][5], career_df['T20I'][5]]]
            wickets = [size if size != special_char else not_played for size in
                       [career_df['Test'][6], career_df['ODI'][6], career_df['T20'][6], career_df['T20I'][6]]]
            bowling_average = [size if size != special_char else not_played for size in
                               [career_df['Test'][7], career_df['ODI'][7], career_df['T20'][7], career_df['T20I'][7]]]
            five_wicket = [size if size != special_char else not_played for size in
                           [career_df['Test'][8], career_df['ODI'][8], career_df['T20'][8], career_df['T20I'][8]]]
            ten_wicket = [size if size != special_char else not_played for size in
                          [career_df['Test'][9], career_df['ODI'][9], career_df['T20'][9], career_df['T20I'][9]]]
            best_bowling = career_df['Test'][10], career_df['ODI'][10], career_df['T20'][10], career_df['T20I'][10]
            ball_label = ['Test', 'ODI', 'T20', 'T20I']
        elif 'LA' in career_df:
            balls_bowled = [not_played if size == special_char else size.replace(',', '') if ',' in size else size for
                            size
                            in [career_df['Test'][5], career_df['ODI'][5], career_df['FC'][5], career_df['LA'][5]]]
            wickets = [size if size != special_char else not_played for size in
                       [career_df['Test'][6], career_df['ODI'][6], career_df['FC'][6], career_df['LA'][6]]]
            bowling_average = [size if size != special_char else not_played for size in
                               [career_df['Test'][7], career_df['ODI'][7], career_df['FC'][7], career_df['LA'][7]]]
            five_wicket = [size if size != special_char else not_played for size in
                           [career_df['Test'][8], career_df['ODI'][8], career_df['FC'][8], career_df['LA'][8]]]
            ten_wicket = [size if size != special_char else not_played for size in
                          [career_df['Test'][9], career_df['ODI'][9], career_df['FC'][9], career_df['LA'][9]]]
            best_bowling = career_df['Test'][10], career_df['ODI'][10], career_df['FC'][10], career_df['LA'][10]
            ball_label = ['Test', 'ODI', 'FC', 'LA']
        elif 'IPL' in career_df:
            balls_bowled = [not_played if size == special_char else size.replace(',', '') if ',' in size else size for
                            size
                            in [career_df['Test'][5], career_df['ODI'][5], career_df['T20I'][5], career_df['IPL'][5]]]
            wickets = [size if size != special_char else not_played for size in
                       [career_df['Test'][6], career_df['ODI'][6], career_df['T20I'][6], career_df['IPL'][6]]]
            bowling_average = [size if size != special_char else not_played for size in
                               [career_df['Test'][7], career_df['ODI'][7], career_df['T20I'][7], career_df['IPL'][7]]]
            five_wicket = [size if size != special_char else not_played for size in
                           [career_df['Test'][8], career_df['ODI'][8], career_df['T20I'][8], career_df['IPL'][8]]]
            ten_wicket = [size if size != special_char else not_played for size in
                          [career_df['Test'][9], career_df['ODI'][9]
        else:
            balls_bowled = [not_played if size == special_char else size.replace(',', '') if ',' in size else size for
                            size
                            in [career_df['Test'][5], career_df['ODI'][5], career_df['FC'][5], career_df['T20I'][5]]]
            wickets = [size if size != special_char else not_played for size in
                       [career_df['Test'][6], career_df['ODI'][6], career_df['FC'][6], career_df['T20I'][6]]]
            bowling_average = [size if size != special_char else not_played for size in
                               [career_df['Test'][7], career_df['ODI'][7], career_df['FC'][7], career_df['T20I'][7]]]
            five_wicket = [size if size != special_char else not_played for size in
                           [career_df['Test'][8], career_df['ODI'][8], career_df['FC'][8], career_df['T20I'][8]]]
            ten_wicket = [size if size != special_char else not_played for size in
                          [career_df['Test'][9], career_df['ODI'][9], career_df['FC'][9], career_df['T20I'][9]]]
            best_bowling = career_df['Test'][10], career_df['ODI'][10], career_df['FC'][10], career_df['T20I'][10]
            ball_label = ['Test', 'ODI', 'FC', 'T20I']

        return balls_bowled, ball_label, bowling_average, wickets, five_wicket, ten_wicket, best_bowling
    else:
        return None


# Catch-Stumpings
def Catch_Stumpings():
    catch = []
    stumpings = []
    not_played = 'Not Played'  # New value for the "-" entries
    special_char = '–'
    career_df = Career()
    if 'T20' in career_df:
        labels = ['Test', 'ODI', 'T20', 'T20I']
        for format_name in labels:
            catch_stumpings = career_df.loc[career_df['Name'] == 'Catches/stumpings', format_name].values
            catch_stumpings = catch_stumpings[0].split('/')
            catch.append({'format': format_name, 'catch': catch_stumpings[0]})
            stumpings.append({'format': format_name, 'stumpings': catch_stumpings[1]})
            catches = [int(item['catch']) if item['catch'] != special_char else not_played for item in catch]
            stumps = [int(item['stumpings']) if item['stumpings'] != special_char else not_played for item in stumpings]
    elif 'LA' in career_df:
        labels = ['Test', 'ODI', 'FC', 'LA']
        for format_name in labels:
            catch_stumpings = career_df.loc[career_df['Name'] == 'Catches/stumpings', format_name].values
            catch_stumpings = catch_stumpings[0].split('/')
            catch.append({'format': format_name, 'catch': catch_stumpings[0]})
            stumpings.append({'format': format_name, 'stumpings': catch_stumpings[1]})
            catches = [int(item['catch']) if item['catch'] != special_char else not_played for item in catch]
            stumps = [int(item['stumpings']) if item['stumpings'] != special_char else not_played for item in stumpings]
    elif 'IPL' in career_df:
        labels = ['Test', 'ODI', 'T20I', 'IPL']
        for format_name in labels:
            catch_stumpings = career_df.loc[career_df['Name'] == 'Catches/stumpings', format_name].values
            catch_stumpings = catch_stumpings[0].split('/')
            catch.append({'format': format_name, 'catch': catch_stumpings[0]})
            stumpings.append({'format': format_name, 'stumpings': catch_stumpings[1]})
            catches = [int(item['catch']) if item['catch'] != special_char else not_played for item in catch]
            stumps = [int(item['stumpings']) if item['stumpings'] != special_char else not_played for item in stumpings]
    else:
        labels = ['Test', 'ODI', 'FC', 'T20I']
        for format_name in labels:
            catch_stumpings = career_df.loc[career_df['Name'] == 'Catches/stumpings', format_name].values
            catch_stumpings = catch_stumpings[0].split('/')
            catch.append({'format': format_name, 'catch': catch_stumpings[0]})
            stumpings.append({'format': format_name, 'stumpings': catch_stumpings[1]})
            catches = [int(item['catch']) if item['catch'] != special_char else not_played for item in catch]
            stumps = [int(item['stumpings']) if item['stumpings'] != special_char else not_played for item in stumpings]

    return catches, stumps, labels


def Awards_Int():
    df_awards = International_Awards()
    if df_awards is None:
        return None
    else:
        # Extracting year and name from Competition column
        df_awards[['Year', 'Competition']] = df_awards['Competition'].str.split(n=1, expand=True)
        df_awards['Year'] = df_awards['Year'].str.extract('(\d{4})')
        df_awards['Year'] = df_awards['Year'].replace(np.nan, 0).astype(int)
        df_awards['Competition'] = df_awards['Competition'].str.strip()
        # Removing year from Award Name column without cutting the name
        df_awards['Award Name'] = df_awards['Award Name'].str.replace('\d{4}', '', regex=True).str.strip()
        # Renaming Award Name column to Host Country
        df_awards.rename(columns={'Award Name': 'Host Country'}, inplace=True)
        # Reordering the columns
        df_awards = df_awards[['Competition', 'Year', 'Host Country', 'Position']]
        # Sort by Year column in ascending order
        df_awards = df_awards.sort_values('Year', ascending=True)
        df_awards = df_awards.reset_index(drop=True)
        return df_awards


def Awards_IPL():
    df_awards = IPL_Awards()
    if df_awards is None:
        return None
    else:
        # Remove year from Competition column without removing the name
        df_awards['Competition'] = df_awards['Competition'].str.replace('\d{4}', '', regex=True).str.strip()
        # Rename Award Name column to Year
        df_awards.rename(columns={'Award Name': 'Year'}, inplace=True)
        # Reordering the columns
        df_awards = df_awards[['Competition', 'Year', 'Position']]
        # Sort by Year column in ascending order
        df_awards = df_awards.sort_values('Year', ascending=True)
        # Reset the index
        df_awards = df_awards.reset_index(drop=True)

        return df_awards


# Title of page
st.markdown("<h1 style='text-align: center; color: black;'>CrickView</h1>", unsafe_allow_html=True)
# Adding a line between Title and content
st.markdown("***")

# Selecting Player name
with st.sidebar:
    with st.spinner("Loading..."):
        time.sleep(1)

    nationality = st.selectbox("Nationality", options=['India', 'Australia', 'England', 'South Africa', 'New Zealand',
                                                       'Pakistan', 'Sri Lanka', 'West Indies', 'Bangladesh',
                                                       'Zimbabwe'])

    if nationality == 'India':
        player_name = st.radio(
            "Player Name",
            options=['MS_Dhoni', 'Sachin_Tendulkar', 'Virat_Kohli', 'Rahul_Dravid', 'Sourav_Ganguly', 'Virender_Sehwag',
                     'Sunil_Gavaskar', 'VVS_Laxman', 'Mohammad_Azharuddin', 'Ravichandran_Ashwin',
                     'Rohit_Sharma', 'Cheteshwar_Pujara', 'Ravindra_Jadeja', 'Ravi_Shastri',
                     'Anil_Kumble', 'Irfan_Pathan']
        )
    elif nationality == 'Australia':
        player_name = st.radio(
            "Player Name",
            options=['Ricky_Ponting', 'Adam_Gilchrist', 'Glenn_McGrath', 'Steve_Smith_(cricketer)',
                     'David_Warner_(cricketer)',
                     'Matthew_Hayden', 'Justin_Langer', 'Brett_Lee', 'Mitchell_Johnson', 'Nathan_Lyon',
                     'Josh_Hazlewood',
                     'Usman_Khawaja', 'Brad_Haddin', 'Shane_Watson', 'Michael Hussey', 'David Boon']

        )
    elif nationality == 'England':
        player_name = st.radio(
            "Player Name",
            options=['Joe_Root', 'Ben_Stokes', 'Stuart_Broad', 'Jos_Buttler', 'Jofra_Archer',
                     'Jonny_Bairstow', 'Chris_Woakes', 'Moeen_Ali', 'Sam_Curran', 'Jason_Roy',
                     'Eoin_Morgan', 'Liam_Plunkett']

        )
    elif nationality == 'South Africa':
        player_name = st.radio(
            "Player Name",
            options=['AB_de_Villiers', 'Kagiso_Rabada', 'Faf_du_Plessis', 'Hashim_Amla',
                     'Dale_Steyn', 'Imran_Tahir', 'Lungi_Ngidi', 'JP_Duminy', 'Aiden_Markram',
                     'Temba_Bavuma', 'Andile_Phehlukwayo', 'Heinrich_Klaasen']

        )
    elif nationality == 'New Zealand':
        player_name = st.radio(
            "Player Name",
            options=['Kane_Williamson', 'Ross_Taylor', 'Trent_Boult', 'Tim_Southee', 'Martin_Guptill',
                     'Colin_de_Grandhomme', 'BJ_Watling', 'Mitchell_Santner', 'Kyle_Jamieson',
                     'Jimmy_Neesham']

        )
    elif nationality == 'Pakistan':
        player_name = st.radio(
            "Player Name",
            options=['Babar_Azam', 'Shaheen_Afridi', 'Sarfaraz_Ahmed', 'Wahab_Riaz', 'Haris_Sohail',
                     'Shoaib_Malik', 'Faheem_Ashraf']

        )
    elif nationality == 'Sri Lanka':
        player_name = st.radio(
            "Player Name",
            options=['Kumar_Sangakkara', 'Mahela_Jayawardene', 'Muttiah_Muralitharan',
                     'Aravinda_de_Silva', 'Tillakaratne_Dilshan', 'Chaminda_Vaas', 'Lasith_Malinga', 'Angelo_Mathews',
                     'Upul_Tharanga', 'Dinesh_Chandimal', 'Thisara_Perera',
                     'Dimuth_Karunaratne', 'Kusal_Perera']

        )
    elif nationality == 'West Indies':
        player_name = st.radio(
            "Player Name",
            options=['Brian_Lara', 'Vivian_Richards', 'Garfield_Sobers', 'Clive_Lloyd',
                     'Michael_Holding', 'Curtly_Ambrose', 'Courtney_Walsh', 'Chris_Gayle',
                     'Shivnarine_Chanderpaul', 'Dwayne_Bravo', 'Jason_Holder', 'Kemar_Roach',
                     'Shai_Hope']

        )
    elif nationality == 'Bangladesh':
        player_name = st.radio(
            "Player Name",
            options=['Shakib_Al_Hasan', 'Tamim_Iqbal', 'Mahmudullah_Riyad', 'Mashrafe_Mortaza',
                     'Mustafizur_Rahman', 'Mohammad_Ashraful', 'Taskin_Ahmed', 'Mehidy_Hasan_Miraz', 'Soumya_Sarkar',
                     'Liton_Das', 'Rubel_Hossain', 'Mosaddek_Hossain', 'Imrul_Kayes']

        )
    else:
        player_name = st.radio(
            "Player Name",
            options=['Andy_Flower', 'Heath_Streak', 'Grant_Flower', 'Tatenda_Taibu', 'Brendan_Taylor',
                     'Elton_Chigumbura', 'Hamilton_Masakadza', 'Prosper_Utseya', 'Craig_Ervine',
                     'Kyle_Jarvis', 'Sikandar_Raza', 'Regis_Chakabva']

        )

# For making Option selection menu
opt_1, opt_2, opt_3, opt_4, opt_5 = st.tabs(["General", "Batting", "Bowling", "Fielding", "Awards"])

with opt_1:
    col_1, col_2 = st.columns([.8, .2])

    with col_2:
        st.markdown("<h4 style='text-align: center; color: black;'>Player Detail</h4>", unsafe_allow_html=True)
        with st.container():
            # Image Display
            image_url = display_player_image(player_name)
            if image_url is None:
                print()
            else:
                st.image(image_url)

        with st.container():
            birthday, birthplace = Born()
            height = Height()
            markdown_content = f"""
                           <div style='
                               background-color: #d3f3cf;
                               padding: 10px;
                               border-radius: 5px;
                               box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                               line-height: 1.5;
                           '>
                               <span style='font-family: Arial, sans-serif;'><strong>Name:</strong></span> {player_name}<br>
                               <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                               <span style='font-family: Arial, sans-serif;'><strong>Birthday Date:</strong></span> {birthday}<br>
                               <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                               <span style='font-family: Arial, sans-serif;'><strong>Birthplace:</strong></span> {birthplace}<br>
                               <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                               <span style='font-family: Arial, sans-serif;'><strong>Height:</strong></span> {height}<br>
                               <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                               <span style='font-family: Arial, sans-serif;'><strong>National Side:</strong></span> {
            General_Info().loc[General_Info()['Label'] == 'National side', 'Information'].values[0]}
                               <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                               <span style='font-family: Arial, sans-serif;'><strong>Role in Team:</strong></span> {
            General_Info()[General_Info()['Label'].str.contains('Role', case=False, na=False)]['Information'].values[0]
            }<br>
                               <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'></div>"""
            st.markdown(markdown_content, unsafe_allow_html=True)

    with col_1:
        with st.container():
            with st.empty():
                st.plotly_chart(Pie_Match_Played())
        st.markdown("---", unsafe_allow_html=True)

        data_1, data_2 = st.columns(2)
        with data_1:
            st.markdown("<h6 style='text-align: center; color: black;'>Debut Match</h6>",
                        unsafe_allow_html=True)
            with st.container():
                # Get the debut dataframe
                Debut = Debut()
                Debut = Debut.reset_index(drop=True)
                # Apply styling to the DataFrame
                styled_Debut = Debut.style \
                    .set_properties(subset=['Debut'], **{'font-weight': 'bold'}) \
                    .set_properties(**{'font-size': '14px'}) \
                    .set_table_styles([
                    {'selector': 'th',
                     'props': [('background-color', '#FFFACD'), ('color', 'black'), ('text-align', 'left'),
                               ('font-weight', 'bold')]},
                    {'selector': 'td',
                     'props': [('background-color', '#FFFFE0'), ('color', 'black'), ('text-align', 'center')]},
                    {'selector': '', 'props': [('border', '1px solid black')]}
                ])
                st.table(styled_Debut)

        with data_2:
            st.markdown("<h6 style='text-align: center; color: black;'>Last Match</h6>",
                        unsafe_allow_html=True)
            with st.container():
                # Get the debut dataframe
                Last = Last()
                # Apply styling to the DataFrame
                styled_Last = Last.style \
                    .set_properties(subset=['Last'], **{'font-weight': 'bold'}) \
                    .set_properties(**{'font-size': '14px'}) \
                    .set_table_styles([
                    {'selector': 'th',
                     'props': [('background-color', '#FFFACD'), ('color', 'black'), ('text-align', 'left'),
                               ('font-weight', 'bold')]},
                    {'selector': 'td',
                     'props': [('background-color', '#FFFFE0'), ('color', 'black'), ('text-align', 'center')]},
                    {'selector': '', 'props': [('border', '1px solid black')]}
                ])
                st.table(styled_Last)

with opt_2:
    run_col_1, run_col_2 = st.columns([.8, .2])

    run_value, run_label, run_average, high_run, cent, half_cent = Pie_Runs_Scored()

    with run_col_1:
        with st.container():
            fig_run = go.Figure(data=[go.Pie(labels=run_label, values=run_value,
                                             marker=dict(colors=['#A2B4DD', '#334572', '#6181C8', '#48609C'],
                                                         line=dict(color='#000000',
                                                                   width=2)))])
            # Customize layout
            fig_run.update_layout(
                font=dict(family='Arial', size=20, color='#333333'),
                legend=dict(
                    title='Legend',
                    orientation='v',
                    x=1, y=0.5,
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='rgba(0, 0, 0, 0.5)',
                    borderwidth=1,
                    font=dict(family='Arial', size=12),
                ),
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                margin=dict(l=50, r=50, t=80, b=50),
                height=500,  # Set the height of the graph (in pixels)
                width=900,  # Set the width of the graph (in pixels)
                title=dict(
                    text='Runs Scored',
                    x=0.5,  # Set the x-position of the title to 0.5 (center)
                    y=0.95,  # Set the y-position of the title
                    xanchor='center',  # Anchor the x-position to the center
                    yanchor='top'  # Anchor the y-position to the top
                )
            )
            # Customize pie chart
            fig_run.update_traces(
                textposition='inside',
                textinfo='percent',
                hoverinfo='label+percent+value',
            )
            st.plotly_chart(fig_run)

        run_1, run_2 = st.columns([.7, .3])

        with run_1:
            with st.container():
                bar_trace = go.Bar(
                    x=run_label,
                    y=run_average,
                    marker=dict(color='#ffa600'),
                )

                # Create a layout
                layout = go.Layout(
                    title='Average Runs',
                    title_x=0.5,
                    xaxis=dict(title='Format', showgrid=True, gridcolor='black'),  # Show x-axis gridlines
                    yaxis=dict(title='Average', showgrid=True, gridcolor='black'),  # Show y-axis gridlines
                    plot_bgcolor='white',  # Set plot background color to light gray
                    margin=dict(l=50, r=50, t=50, b=50),  # Set margins around the plot
                    bargap=0.2,  # Set gap between bars
                    bargroupgap=0.1,  # Set gap between bar groups
                )

                # Create a Figure object
                average = go.Figure(data=[bar_trace], layout=layout)

                # Display the figure
                st.plotly_chart(average)
        with run_2:
            st.write("")  # Add empty space
            st.write("")  # Add empty space
            st.write("")  # Add empty space
            st.write("")  # Add empty space
            st.write("")  # Add empty space
            st.write("")  # Add empty space

            with st.container():
                markdown_content = f"""
                    <div style='
                        background-color: #d3f3cf;
                        padding: 5px;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        line-height: 1.5;
                        width: 200px;  /* Decrease the width of the box */
                    '>
                        <h3 style='text-align: center; font-family: Arial, sans-serif;'>Highest Runs</h3>  <!-- Add the title -->
                        <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                        <span style='font-family: Arial, sans-serif;'><strong>{run_label[0]}:</strong></span> {high_run[0]}<br>
                        <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                        <span style='font-family: Arial, sans-serif;'><strong>{run_label[1]}:</strong></span> {high_run[1]}<br>
                        <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                        <span style='font-family: Arial, sans-serif;'><strong>{run_label[2]}:</strong></span> {high_run[2]}<br>
                        <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                        <span style='font-family: Arial, sans-serif;'><strong>{run_label[3]}:</strong></span> {high_run[3]}<br>
                        <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'></div>"""
                st.markdown(markdown_content, unsafe_allow_html=True)

    with run_col_2:
        with st.container():
            # Create pie chart
            fig_cent = go.Figure(data=[
                go.Pie(labels=run_label, values=cent, marker=dict(colors=['#003f5c', '#58508d', '#bc5090', '#ff6361'],
                                                                  line=dict(color='#000000',
                                                                            width=2)))])
            # Customize layout
            fig_cent.update_layout(
                font=dict(family='Arial', size=16, color='#333333'),
                legend=dict(
                    title='Legend',
                    orientation='v',
                    x=1, y=0.5,
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='rgba(0, 0, 0, 0.5)',
                    borderwidth=1,
                    font=dict(family='Arial', size=12),
                ),
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                margin=dict(l=50, r=50, t=80, b=50),
                height=300,  # Set the height of the graph (in pixels)
                width=300,  # Set the width of the graph (in pixels)
                title=dict(
                    text='Total Centuries(100) per Format',
                    x=0.5,  # Set the x-position of the title to 0.5 (center)
                    y=0.95,  # Set the y-position of the title
                    xanchor='center',  # Anchor the x-position to the center
                    yanchor='top'  # Anchor the y-position to the top
                )
            )
            # Customize pie chart
            fig_cent.update_traces(
                textposition='inside',
                textinfo='percent',
                hoverinfo='label+percent+value',
            )
            st.plotly_chart(fig_cent)
        st.markdown("---", unsafe_allow_html=True)  # Line Break

        with st.container():
            fig_half_cent = go.Figure(data=[go.Pie(labels=run_label, values=half_cent,
                                                   marker=dict(colors=['#ED5F5F', '#8ED2E7', '#ECE44C', '#E9A555'],
                                                               line=dict(color='#000000',
                                                                         width=2)))])
            fig_half_cent.update_layout(
                font=dict(family='Arial', size=16, color='#333333'),
                legend=dict(
                    title='Legend',
                    orientation='v',
                    x=1, y=0.5,
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='rgba(0, 0, 0, 0.5)',
                    borderwidth=1,
                    font=dict(family='Arial', size=12),
                ),
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                margin=dict(l=50, r=50, t=80, b=50),
                height=300,  # Set the height of the graph (in pixels)
                width=300,  # Set the width of the graph (in pixels)
                title=dict(
                    text='Total Half-Centuries(50) per Format',
                    x=0.5,  # Set the x-position of the title to 0.5 (center)
                    y=0.95,  # Set the y-position of the title
                    xanchor='center',  # Anchor the x-position to the center
                    yanchor='top'  # Anchor the y-position to the top
                )
            )
            # Customize pie chart
            fig_half_cent.update_traces(
                textposition='inside',
                textinfo='percent',
                hoverinfo='label+percent+value',
            )
            st.plotly_chart(fig_half_cent)

        st.markdown("---", unsafe_allow_html=True)  # Line Break

with opt_3:
    ball_col_1, ball_col_2 = st.columns([.8, .2])

    if Pie_Balls() is not None:
        balls_bowled, ball_label, bowling_average, wickets, five_wicket, ten_wicket, best_bowling = Pie_Balls()

        with ball_col_1:
            with st.container():
                fig_ball = go.Figure(data=[go.Pie(labels=ball_label, values=balls_bowled,
                                                  marker=dict(colors=["#ea5545", "#f46a9b", "#ef9b20", "#edbf33"],
                                                              line=dict(color='#000000',
                                                                        width=2)))])
                # Customize layout
                fig_ball.update_layout(
                    font=dict(family='Arial', size=20, color='#333333'),
                    legend=dict(
                        title='Legend',
                        orientation='v',
                        x=1, y=0.5,
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        bordercolor='rgba(0, 0, 0, 0.5)',
                        borderwidth=1,
                        font=dict(family='Arial', size=12),
                    ),
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    margin=dict(l=50, r=50, t=80, b=50),
                    height=500,  # Set the height of the graph (in pixels)
                    width=900,  # Set the width of the graph (in pixels)
                    title=dict(
                        text='Balls Bowled',
                        x=0.5,  # Set the x-position of the title to 0.5 (center)
                        y=0.95,  # Set the y-position of the title
                        xanchor='center',  # Anchor the x-position to the center
                        yanchor='top'  # Anchor the y-position to the top
                    )
                )
                # Customize pie chart
                fig_ball.update_traces(
                    textposition='inside',
                    textinfo='percent',
                    hoverinfo='label+percent+value',
                )
                st.plotly_chart(fig_ball)

            ball_1, ball_2 = st.columns([.7, .3])

            with ball_1:
                with st.container():
                    bar_trace = go.Bar(
                        x=ball_label,
                        y=bowling_average,
                        marker=dict(color="#9b19f5"),
                    )

                    # Create a layout
                    layout = go.Layout(
                        title='Bowling Average',
                        title_x=0.5,
                        xaxis=dict(title='Format', showgrid=True, gridcolor='black'),  # Show x-axis gridlines
                        yaxis=dict(title='Average', showgrid=True, gridcolor='black'),  # Show y-axis gridlines
                        plot_bgcolor='white',  # Set plot background color to light gray
                        margin=dict(l=50, r=50, t=50, b=50),  # Set margins around the plot
                        bargap=0.2,  # Set gap between bars
                        bargroupgap=0.1,  # Set gap between bar groups
                    )

                    # Create a Figure object
                    average = go.Figure(data=[bar_trace], layout=layout)

                    # Display the figure
                    st.plotly_chart(average)

            with ball_2:
                st.write("")  # Add empty space
                st.write("")  # Add empty space
                st.write("")  # Add empty space
                st.write("")  # Add empty space
                st.write("")  # Add empty space
                st.write("")  # Add empty space

                with st.container():
                    markdown_content = f"""
                            <div style='
                                background-color: #d3f3cf;
                                padding: 5px;
                                border-radius: 5px;
                                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                                line-height: 1.5;
                                width: 200px;  /* Decrease the width of the box */
                            '>
                                <h3 style='text-align: center; font-family: Arial, sans-serif;'>Best Bowling Figure</h3>  <!-- Add the title -->
                                <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                                <span style='font-family: Arial, sans-serif;'><strong>{ball_label[0]}:</strong></span> {best_bowling[0]}<br>
                                <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                                <span style='font-family: Arial, sans-serif;'><strong>{ball_label[1]}:</strong></span> {best_bowling[1]}<br>
                                <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                                <span style='font-family: Arial, sans-serif;'><strong>{ball_label[2]}:</strong></span> {best_bowling[2]}<br>
                                <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'>
                                <span style='font-family: Arial, sans-serif;'><strong>{ball_label[3]}:</strong></span> {best_bowling[3]}<br>
                                <hr style='border: 1px solid #9ed9a7; margin: 5px 0;'></div>"""
                    st.markdown(markdown_content, unsafe_allow_html=True)

        with ball_col_2:
            with st.container():
                # Create pie chart
                fig_wickets = go.Figure(data=[
                    go.Pie(labels=ball_label, values=wickets,
                           marker=dict(colors=["#e60049", "#0bb4ff", "#50e991", "#e6d800"],
                                       line=dict(color='#000000',
                                                 width=2)))])
                # Customize layout
                fig_wickets.update_layout(
                    font=dict(family='Arial', size=16, color='#333333'),
                    legend=dict(
                        title='Legend',
                        orientation='v',
                        x=1, y=0.5,
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        bordercolor='rgba(0, 0, 0, 0.5)',
                        borderwidth=1,
                        font=dict(family='Arial', size=12),
                    ),
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    margin=dict(l=50, r=50, t=80, b=50),
                    height=300,  # Set the height of the graph (in pixels)
                    width=300,  # Set the width of the graph (in pixels)
                    title=dict(
                        text='Total Wickets per Format',
                        x=0.5,  # Set the x-position of the title to 0.5 (center)
                        y=0.95,  # Set the y-position of the title
                        xanchor='center',  # Anchor the x-position to the center
                        yanchor='top'  # Anchor the y-position to the top
                    )
                )
                # Customize pie chart
                fig_wickets.update_traces(
                    textposition='inside',
                    textinfo='percent',
                    hoverinfo='label+percent+value',
                )
                st.plotly_chart(fig_wickets)
            st.markdown("---", unsafe_allow_html=True)  # Line Break
            with st.container():
                fig_five_wicket = go.Figure(data=[go.Pie(labels=ball_label, values=five_wicket,
                                                         marker=dict(
                                                             colors=["#00b7c7", "#5ad45a", "#8be04e", "#ebdc78"],
                                                             line=dict(color='#000000',
                                                                       width=2)))])
                fig_five_wicket.update_layout(
                    font=dict(family='Arial', size=16, color='#333333'),
                    legend=dict(
                        title='Legend',
                        orientation='v',
                        x=1, y=0.5,
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        bordercolor='rgba(0, 0, 0, 0.5)',
                        borderwidth=1,
                        font=dict(family='Arial', size=12),
                    ),
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    margin=dict(l=50, r=50, t=80, b=50),
                    height=300,  # Set the height of the graph (in pixels)
                    width=300,  # Set the width of the graph (in pixels)
                    title=dict(
                        text='5 wickets in an Innings',
                        x=0.5,  # Set the x-position of the title to 0.5 (center)
                        y=0.95,  # Set the y-position of the title
                        xanchor='center',  # Anchor the x-position to the center
                        yanchor='top'  # Anchor the y-position to the top
                    )
                )
                # Customize pie chart
                fig_five_wicket.update_traces(
                    textposition='inside',
                    textinfo='percent',
                    hoverinfo='label+percent+value',
                )
                st.plotly_chart(fig_five_wicket)

            st.markdown("---", unsafe_allow_html=True)  # Line Break

            with st.container():
                fig_ten_wicket = go.Figure(data=[go.Pie(labels=ball_label, values=ten_wicket,
                                                        marker=dict(colors=["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe"],
                                                                    line=dict(color='#000000',
                                                                              width=2)))])
                fig_ten_wicket.update_layout(
                    font=dict(family='Arial', size=16, color='#333333'),
                    legend=dict(
                        title='Legend',
                        orientation='v',
                        x=1, y=0.5,
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        bordercolor='rgba(0, 0, 0, 0.5)',
                        borderwidth=1,
                        font=dict(family='Arial', size=12),
                    ),
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    margin=dict(l=50, r=50, t=80, b=50),
                    height=300,  # Set the height of the graph (in pixels)
                    width=300,  # Set the width of the graph (in pixels)
                    title=dict(
                        text='10 wickets in a Match',
                        x=0.5,  # Set the x-position of the title to 0.5 (center)
                        y=0.95,  # Set the y-position of the title
                        xanchor='center',  # Anchor the x-position to the center
                        yanchor='top'  # Anchor the y-position to the top
                    )
                )
                # Customize pie chart
                fig_ten_wicket.update_traces(
                    textposition='inside',
                    textinfo='percent',
                    hoverinfo='label+percent+value',
                )
                st.plotly_chart(fig_ten_wicket)

with opt_4:
    field_col_1, field_col_2 = st.columns([.5, .5])
    catches, stumps, labels = Catch_Stumpings()

    with field_col_1:
        with st.container():
            # Create pie chart
            fig_catch = go.Figure(
                data=[go.Pie(labels=labels, values=catches,
                             marker=dict(colors=["#a6d75b", "#c9e52f", "#d0ee11", "#d0f400"],
                                         line=dict(color='#000000',
                                                   width=2)))])
            # Customize layout
            fig_catch.update_layout(
                font=dict(family='Arial', size=20, color='#333333'),
                legend=dict(
                    title='Legend',
                    orientation='v',
                    x=1, y=0.5,
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='rgba(0, 0, 0, 0.5)',
                    borderwidth=1,
                    font=dict(family='Arial', size=12),
                ),
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                margin=dict(l=50, r=50, t=80, b=50),
                height=500,  # Set the height of the graph (in pixels)
                width=600,  # Set the width of the graph (in pixels)
                title=dict(
                    text='Total Catches',
                    x=0.5,  # Set the x-position of the title to 0.5 (center)
                    y=0.95,  # Set the y-position of the title
                    xanchor='center',  # Anchor the x-position to the center
                    yanchor='top'  # Anchor the y-position to the top
                )
            )
            # Customize pie chart
            fig_catch.update_traces(
                textposition='inside',
                textinfo='percent',
                hoverinfo='label+percent+value',
            )
            st.plotly_chart(fig_catch)

        st.markdown("---", unsafe_allow_html=True)  # Line Break

    with field_col_2:
        with st.container():
            bar_trace = go.Bar(
                x=labels,
                y=stumps,
                marker=dict(color="#009fff"),
            )

            # Create a layout
            fig_stumps = go.Layout(
                title='Total Stumpings',
                title_x=0.5,
                xaxis=dict(title='Format', showgrid=True, gridcolor='black'),  # Show x-axis gridlines
                yaxis=dict(title='Stumps', showgrid=True, gridcolor='black'),  # Show y-axis gridlines
                plot_bgcolor='white',  # Set plot background color to light gray
                margin=dict(l=50, r=50, t=50, b=50),  # Set margins around the plot
                bargap=0.2,  # Set gap between bars
                bargroupgap=0.1,  # Set gap between bar groups
            )

            # Create a Figure object
            fig_stumps = go.Figure(data=[bar_trace], layout=fig_stumps)

            # Display the figure
            st.plotly_chart(fig_stumps)

with opt_5:
    award_col_1, award_col_2 = st.columns([.7, .3])

    with award_col_1:
        st.markdown("<h6 style='text-align: center; color: black;'>International Awards</h6>",
                    unsafe_allow_html=True)
        with st.container():
            # Get the debut dataframe
            awards = Awards_Int()

            if awards is None:
                print()
            else:
                def color(position):
                    colors = 'lightgreen' if position == 'Winner' else 'orange' if position == 'Runner-up' else 'white'
                    return f'background-color: {colors}'


                # Apply styling to the DataFrame
                styled_awards = awards.style \
                    .set_properties(subset=['Competition'], **{'font-weight': 'bold'}) \
                    .set_properties(**{'font-size': '14px'}) \
                    .background_gradient(subset=['Year'], cmap='Blues')

                # Apply cell color based on Position
                styled_awards = styled_awards.applymap(color, subset=['Position'])

                # Convert the DataFrame to HTML
                html_table = styled_awards.to_html(index=False)

                styled_html = f"""
                            <style>
                            table {{
                                margin-left: auto;
                                margin-right: auto;
                            }}
                            th, td {{
                                border: 1px solid black;
                                padding: 8px;
                            }}
                            th {{
                                background-color: #FFFFFF;
                                color: black;
                                text-align: left;
                                font-weight: bold;
                            }}
                            td {{
                                background-color: #FFFFFF;
                                color: black;
                                text-align: center;
                            }}
                            tr:nth-child(odd) {{
                                background-color: white;
                            }}
                            table td {{
                                border: 1px solid #666666;
                            }}
                            {html_table}
                            """

                # Display the styled HTML table
                st.write(styled_html, unsafe_allow_html=True)

    with award_col_2:
        st.markdown("<h6 style='text-align: center; color: black;'>IPL Awards</h6>",
                    unsafe_allow_html=True)
        with st.container():
            # Get the debut dataframe
            awards_IPL = Awards_IPL()

            if awards_IPL is None:
                print()
            else:
                def color(position):
                    colors = 'lightgreen' if position == 'Winner' else 'orange' if position == 'Runner-up' else 'white'
                    return f'background-color: {colors}'


                # Apply styling to the DataFrame
                styled_awards_ipl = awards_IPL.style \
                    .set_properties(subset=['Competition'], **{'font-weight': 'bold'}) \
                    .set_properties(**{'font-size': '14px'}) \
                    .background_gradient(subset=['Year'], cmap='Blues')

                # Apply cell color based on Position
                styled_awards_ipl = styled_awards_ipl.applymap(color, subset=['Position'])

                # Convert the DataFrame to HTML
                html_table = styled_awards_ipl.to_html(index=False)

                # Apply CSS styles to the HTML table
                styled_html_IPL = f"""
                           <style>
                           th, td {{
                               border: 1px solid black;
                               padding: 8px;
                           }}
                           th {{
                               background-color: #FFFFFF;
                               color: black;
                               text-align: left;
                               font-weight: bold;
                           }}
                           td {{
                               background-color: #FFFFFF;
                               color: black;
                               text-align: center;
                           }}
                           tr:nth-child(odd) {{
                               background-color: white;
                           }}
                           table td {{
                               border: 1px solid #666666;
                           }}
                           {html_table}
                           """

                # Display the styled HTML table
                st.write(styled_html_IPL, unsafe_allow_html=True)
