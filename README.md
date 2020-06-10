# Predicting NBA Salaries to Help Players and Agents

#### Authors: Markell Jones-Francis [(GitHub)](https://github.com/mjones-francis),  [(LinkedIn)](https://www.linkedin.com/in/markell-jones-francis/)

## Problem Statement

NBA Salary negotiations are complicated. They require a multitude of stakeholders, each with their own agendas and interests, coming together to agree on a contract with a set number of years and often a guaranteed salary. Players and agents look to maximize their earnings by signing for the most amount they can, while General Managers will often try to keep numbers manageable so they have room to continue signing more players without exceeding the league-mandated salary cap.  

The issue at the heart of these negotiations is the player's statistical performance. While outside factors such as interpersonal relationships and star power play a role, the statistics that a player is able to record while playing and their ability to help their team matter more than any other factor when negotiating a contract. To streamline this process for players and agents, we will be building a machine learning model using each player's basic, advanced, and team statistics in order to accurately predict their salary. To do this we will be using Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, and Neural Network models and, given the scale of NBA contracts, will be looking to make predictions that are accurate within \\$2,000,000 per year, using RMSE as our metric.  

We will be using data from the 2016-17 to the 2018-19 seasons for these projections, because the 2016-17 season saw the NBA salary cap increase to \\$94M, from \\$70M the season before, as a result of the NBA's then newly signed television deal which brought in additional revenue. This cap increase resulted in salary increases across the board for NBA players, so much so that any salaries 2016-17 bear little comparative value when evaluating statistical performance. We are also ending the analysis at the 2018-19 season as that is the most recent completed NBA season.  

With this tool we hope to assist agents find the most fair salaries for the players they represent, and determine which teams are most likely pay their clients the salary that they deserve.
 
## Executive Summary
Before we could any of our analysis or modeling, we first had to gather our data. After much searching, we decided to use Basketball Reference for our Basic and Advanced Statistics, ESPN for our Team Statistics, and Hoops Hype for our salary data. One hurdle we came across during the data gathering phase, was that several potential sources of useful data, had said data commented out in source code of the web page - though whether this was an oversight or an intentional move to block scraping is unclear. Once we finalized our data sources, we used the BeautifulSoup4 Python package to scrape the tables of the appropriate web pages and import the information and combined each data source into one master dataframe, merged on Player, Team, and Season. The full data gathering process can be found in the Data Gathering folder of this repo. 
  
Once our data hand was in hand and cleaned we were able to begin our analysis. Our analysis proved some of our initial assumptions to be true, it also offered many surprises, and this all began by visualizing the correlation between our various features and our target variable, Salary. We knew before the analysis began that scoring and statistics related to scoring would be highly correlated to Salary - the NBA is an on offense-driven league after all. But what caught us off-guard was the relatively limited correlation that many advanved stats had to Salary. For all the discourse surrounding the modern NBA and the importance of analytics, many of the advanced statistics basketball afficianados obsess over had little correlation to how much a team was willing to pay a player.  But we also understood that correlation is not the same as causation so were delved deeper.  
  
After additional analysis we began our initial attempt at modeling which could only be described as an absolute failure. We tested dozens of combinations on each our models and each one returned testing predictions that were off by an average at least $6 Million. We realized that a large part of this was due to younger players whose performance would suggest a high contract value, but whose earnings were limited by NBA policy regarding recently drafted players. While we knew before we began that the intricacies of NBA contracts would have an effect on our analysis and our models' ability to generate predictions, we had hoped that these would be accounted for by the relationships between various features in our data. We were wrong.  
  
We then went back to our analysis in order to find a way to account for these changes. Understanding the importance that age had on a players salary, we knew we had to find a way to make our model understand that younger players were limited to a certain salary range, but also that players are not playerd more simply because they are older. We opted to do this by K-Means Clustering in order segment our data set into distinct groups of players, so our models could more easily distinguish between high and low earners. The end result was a final model that, while not perfect, offered a significant improvement over our initial efforts allowed us to gain actionable insights moving forward.
## Data Sources:
[NBA Basic Player Statistics](https://www.basketball-reference.com/leagues/NBA_2017_per_game.html)
[NBA Advanced Player Statistics](https://www.basketball-reference.com/leagues/NBA_2017_advanced.html)
[NBA Team Statistics](https://www.espn.com/nba/stats/team/2017/season/2017/seasontype/2)
[NBA Salary Data](https://hoopshype.com/salaries/players/2016-2017/)

### Data Dictionary 
|**Feature Name**|**Description**|
|:---|:---|
|player|Player Name|
|pos|Player's Position|
|age|Player's Age|
|team_id|Player's Team|
|g|Games Played|
|gs|Games Started|
|mp_per_g|Minutes Played Per Game|
|fg_per_g|Total Field Goals Made Per Game|
|fga_per_g|Total Field Goal Attempts Per Game|
|fg_pct|Field Goal Percentage|
|fg3_per_g|3-Point Field Goals Made Per Game|
|fg3_a_per_g|3-Point Field Goal Attempts Per Game|
|fg3_pct|3-Point Field Goal Make Percentage|
|fg2_per_g|2-Point Field Goals Made Per Game|
|fg2a_per_g|2-Point Field Goal Attempts Per Game|
|fg2_pct|2-Point Field Goal Make Percentage|
|efg_pct|Effective Field Goal Percentage|
|ft_per_g|Free Throws Made Per Game|
|fta_per_g|Free Throw Attempts Per Game|
|ft_pct|Free Throw Make Percentage|
|orb_per_g|Offensive Rebounds Per Game|
|drb_per_g|Defensive Rebounds Per Game|
|trb_per_g|Total Rebounds Per Game|
|ast_per_g|Assists Per Game|
|stl_per_g|Steals Per Game|
|blk_per_g|Blocks Per Game|
|tov_per_g|Turnovers Per Game|
|pf_per_g|Personal Fouls Per Game|
|pts_per_g|Points Per Game|
|season|Season (Calendar Year in Which season ended)|
|mp|Total Minutes Played|
|per|Player Efficiency Rating (A measure of per-minute production standardized such that the league average is 15)|
|ts_pct|True Shooting Percentage (A measure of shooting efficiency that takes into account 2-point field goals, 3-point field goals, and free throws)|
|fg3a_per_fga_pct|3-Point Attempt Rate|
|fta_per_fga_pct|Free Throw Attempt Rate|
|orb_pct|Offensive Rebound Percentage|
|drb_pct|Defensive Rebound Percentage|
|trb_pct|Total Rebound Percentage|
|ast_pct|Assist Percentage (An estimate of the percentage of teammate field goals a player assisted while he was on the floor)|
|stl_pct|Steal Percentage (An estimate of the percentage of opponent possessions that end with a steal by the player while he was on the floor)|
|blk_pct|Block Percentage (An estimate of the percentage of opponent two-point field goal attempts blocked by the player while he was on the floor)|
|Turnover Percentage|An estimate of turnovers committed per 100 plays|
|usg_pct|Usage Percentage (An estimate of the percentage of team plays used by a player while he was on the floor)|
|ows|Offensive Win Shares (An estimate of the number of wins contributed by a player due to his offense)|
|dws|Defensive Win Shares (An estimate of the number of wins contributed by a player due to his defense)|
|ws|Win Shares (An estimate of the number of wins contributed by a playe)|
|ws_per_48|Win Shares Per 48 Minutes (An estimate of the number of wins contributed by a player per 48 minutes (league average is approximately .10)|
|obpm|Offensive Box Plus/Minus (A box score estimate of the offensive points per 100 possessions a player contributed above a league-average player, translated to an average team)|
|dbpm|Defensive Box Plus/Minus (Defensive Box Plus/Minus</b><br>A box score estimate of the defensive points per 100 possessions a player contributed above a league-average player, translated to an average team)|
|bpm|Box Plus/Minus (A box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team)|
|vorp|Value Over Replacement Player (A box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season. Multiply by 2.70 to convert to wins over replacement)|
|salary|Player's Salary|
|team_PTS|Points Scored Per Game by the player's team|
|team_FGM|Field Goals Made Per Game by the player's team|
|team_FGA|Field Goals Attempted Per Game by the player's team|
|team_3PM|3-Point Field Goals Made Per Game by the player's team|
|team_3PA|3-Point Field Goals Attempted Per Game by the player's team|
|team_FTM|Free Throws Made Per Game by the player's team|
|team_FTA|Free Throws Attempted Per Game by the player's team|
|team_REB|Rebounds Per Game by the player's team|
|team_AST|Assists Per Game by the player's team|

## Conclusion
What we have found through the exploration of our data and the modeling process is that there is far more that influences an NBA player's salary than just raw statistical output, and the two sides have much more to discuss when at the negotiation table. However, when factors that are beyond the scope of on-court performance are limited and NBA contract policy does not artificially deflate a player's earning potential, machine learning can be a valuable tool to get an a rough estimate of what an NBA player should be paid, based on what he can contribute on the court over the course of a season. Thus, we believe that machine learning can be used by players and agents to set a starting point for salary negotiations and by General Managers to evaluate their own performance on player signings on a player-by-player basis.  
  
In the future, we hope to improve the depth of our analysis and our model's predictive capabilities by gathering more data and finding ways to add additional features that fully capture the context of each player's individual situation as well as the broader trends around the NBA. Eventually we would like to create a tool that will not only establish whether each player is generating a statistical output worthy of their contract, but that will also be able to accurately predict the value of future contracts based on past performance.

## References

[Salary Cap Information](http://www.cbafaq.com/salarycap.htm#Q16)  
[SuperMax Contract Effects](https://www.nbcsports.com/washington/wizards/explained-what-nba-supermax-contract-and-how-does-it-work#:~:text=What%20is%20a%20supermax%20contract,escalation%20in%20each%20subsequent%20year.)  
[Gordon Hayward's Injury](https://theundefeated.com/features/gordon-haywards-gruesome-injury-was-no-time-for-jokes-and-hot-takes/#:~:text=Hayward's%20injury%2C%20which%20has%20been,Paul%20George's%20tibia%2Dfibula%20fracture)  
[Draymond Green's Importance](https://www.goldenstateofmind.com/2020/5/5/21247371/greens-impact-on-basketball-equal-to-currys)
