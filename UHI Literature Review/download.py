import pandas as pd
import webbrowser

papers = pd.read_excel('Urban_heat_island_air_temp_only.xls',
                       sheet_name = 'savedrecs',
                       header = 28)

for title, doi in zip(papers.Title, papers.DOI):

    if pd.isna(doi): # if there's no DOI, search in Google Scholar:
        url = 'https://scholar.google.de/scholar?hl=de&as_sdt=0%2C5&q='+title

        # This was AAU Primo, but didn't work that well:
        #url = 'https://aub-primo.hosted.exlibrisgroup.com/primo_library/libweb/action/search.do?amp=&vl(57399062UI1)=all_items&indx=1&fn=search&dscnt=0&vl(1UIStartWith0)=contains&initializeIndex=true&vid=desktop&ct=search&vl(57399064UI0)=any&tab=default_tab&institute=&dum=true&vl(freeText0)='+title+'&dstmp=1542746277371&fromLogin=true'
    else: # otherwise just follow the redirect from the DOI resolver
        url = 'httP://dx.doi.org/'+str(doi)

    webbrowser.open(url)
