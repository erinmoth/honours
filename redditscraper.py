#scrape title data from reddit
import pandas as pd
from psaw import PushshiftAPI
import datetime as dt
import sys
api = PushshiftAPI()


start_epoch=int(dt.datetime(2019, 1, 1).timestamp())
arguments = sys.argv
outname = "reddit_dataset_" + arguments[1] +".csv"
subreddit_list = arguments[2:]

#subreddit_list=  ['3DS', '2007scape', 'battlefield_4', 'boardgames', 'CasualPokemonTrades', 'Civcraft', 'ClashOfClans', 'CoDCompetitive', 'csgobetting', 'darksouls', 'DarkSouls2', 'DestinyTheGame', 'Diablo', 'DnD', 'DotA2', 'dragonage', 'elderscrollsonline', 'Eve', 'FIFA', 'GameDeals', 'gamegrumps', 'Games', 'gaming', 'GrandTheftAutoV', 'Guildwars2', 'halo', 'hearthstone', 'KerbalSpaceProgram', 'leagueoflegend', 'magicTCG', 'Minecraft', 'oculus', 'pathofexile', 'pcgaming', 'pcmasterrace', 'pokemon', 'PokemonPlaza', 'pokemontrades', 'PotterPlayRP', 'PS4', 'RandomActsOfGaming', 'roosterteeth', 'runescape', 'ShinyPokemon', 'skyrim', 'smashbros', 'starcraft', 'Steam', 'SteamGameSwap', 'tf2', 'titanfall', 'twitchplayspokemon', 'Warthunder', 'wow', 'xboxone']

title_list = []

for sub in subreddit_list:
    posts = list(api.search_submissions(after=start_epoch,
                            subreddit=sub,
                            filter=['url','author', 'title', 'subreddit'],
                            limit=50000))
    
    for post in posts:
        #only keeping title information, discard the rest
        title_list.append(post.title)
       
    print(sub, 'completed; ', end='')
    print('total', len(title_list), 'posts have been scraped')

df = pd.DataFrame({ 
                   'Title':title_list
                   
                  })
df.to_csv(outname, index = False)
