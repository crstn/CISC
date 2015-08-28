def main():
    import os
    import pandas as pd
    
    UN_DESA = pd.read_csv("J:/Data/DESA/WTP2014.csv")
    indexed_UN_DESA = UN_DESA.set_index("Country Code")
    
    for index, row in indexed_UN_DESA.iterrows():    
        os.chdir("J:/LIVE")
        new_folder = str("Output_"+str(index))
        os.makedirs(new_folder)

if __name__ == '__main__':
    main()
