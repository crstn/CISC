import bibtexparser
import pandas as pd
import Levenshtein
import webbrowser

# if this is set to true, the script will opena browser window to search for
# each of the papers where no match is found
download = True

with open('UHI.bib') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)


papers = pd.read_excel('Urban_heat_island_air_temp_only.xls',
                       sheet_name = 'savedrecs',
                       header = 28)

nodoi_xls = 0
nodoi_bib = 0
matches = 0
entries = 0

for title_xls, xls_doi in zip(papers.Title, papers.DOI):

    entries = entries + 1
    match = False


    if pd.isna(xls_doi):
        nodoi_xls = nodoi_xls + 1

        # compare titles:
        for entry in bib_database.entries:

            if 'title' in entry:

                ratio = Levenshtein.ratio(str(entry['title']).lower(),
                                          str(title_xls).lower())
                if ratio > 0.7:
                    print(entry['title'])
                    print(title_xls)
                    print(ratio)
                    match = True
    else:

        for entry in bib_database.entries:

            if 'doi' in entry:
                if entry['doi'] == xls_doi:

                    match = True
                    matches = matches + 1
            else:
                nodoi_bib = nodoi_bib + 1


    # if no match is found, opena  browser window to download the paper:
    if not match:
        if pd.isna(xls_doi): # if there's no DOI, search in Google Scholar:
            url = 'https://scholar.google.de/scholar?hl=de&as_sdt=0%2C5&q='+title_xls

            # This was AAU Primo, but didn't work that well:
            #url = 'https://aub-primo.hosted.exlibrisgroup.com/primo_library/libweb/action/search.do?amp=&vl(57399062UI1)=all_items&indx=1&fn=search&dscnt=0&vl(1UIStartWith0)=contains&initializeIndex=true&vid=desktop&ct=search&vl(57399064UI0)=any&tab=default_tab&institute=&dum=true&vl(freeText0)='+title+'&dstmp=1542746277371&fromLogin=true'
        else: # otherwise just follow the redirect from the DOI resolver
            url = 'httP://dx.doi.org/'+str(xls_doi)

        webbrowser.open(url)

print("matches:",str(matches),"out of",str(entries))
print("nodoi in bib:",str(nodoi_bib))
print("nodoi in xls:",str(nodoi_xls))
