from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
import re


class MyCategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        
        def huoneisto_puhdistaja(data):

            # Some have living rooms marked seperately so taking that into account

            huoneiden_lkm = np.zeros(len(data["Huoneisto"]))
            data["Huoneistotyyppi"] = data["Huoneistotyyppi"].fillna("puuttuva")
            data["Huoneisto"] = data["Huoneisto"].fillna("puuttuva")

            # If marked as yksö => 1
            huoneiden_lkm[data["Huoneistotyyppi"].str.match("^.*Yksiö.*$", case=False)] = 1
            huoneiden_lkm[data["Huoneistotyyppi"].str.match("^.*Kaksi.*$", case=False)] = 2
            huoneiden_lkm[data["Huoneistotyyppi"].str.match("^.*Kolme.*$", case=False)] = 3

            oh_talot = data["Huoneisto"].str.match("^.*oh.*$", case=False)

            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(2|2h) *- *(3|3h).*$", case=False), oh_talot)] = 4
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(3|3h) *- *(4|4h).*$", case=False), oh_talot)] = 5
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(4|4h) *- *(5|5h).*$", case=False), oh_talot)] = 6
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(5|5h) *- *(6|6h).*$", case=False), oh_talot)] = 7
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(6|6h) *- *(7|7h).*$", case=False), oh_talot)] = 8
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(7|7h) *- *(8|8h).*$", case=False), oh_talot)] = 9
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(8|8h) *- *(9|9h).*$", case=False), oh_talot)] = 10
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(9|9h) *- *(10|10h).*$", case=False), oh_talot)] = 11


            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(2|2h) *- *(3|3h).*$", case=False), ~oh_talot)] = 3
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(3|3h) *- *(4|4h).*$", case=False), ~oh_talot)] = 4
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(4|4h) *- *(5|5h).*$", case=False), ~oh_talot)] = 5
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(5|5h) *- *(6|6h).*$", case=False), ~oh_talot)] = 6
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(6|6h) *- *(7|7h).*$", case=False), ~oh_talot)] = 7
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(7|7h) *- *(8|8h).*$", case=False), ~oh_talot)] = 8
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(8|8h) *- *(9|9h).*$", case=False), ~oh_talot)] = 9
            huoneiden_lkm[np.logical_and(data["Huoneisto"].str.match("^.*(9|9h) *- *(10|10h).*$", case=False), ~oh_talot)] = 10

            #what does 2-4h mean. But I'll go with the higher value
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(2|2h) *- *(4|4h).*$", case=False)] = 4
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(3|3h) *- *(5|5h).*$", case=False)] = 5
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(4|4h) *- *(6|6h).*$", case=False)] = 6
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(5|5h) *- *(7|7h).*$", case=False)] = 7
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(6|6h) *- *(8|8h).*$", case=False)] = 8
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(7|7h) *- *(9|9h).*$", case=False)] = 9
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(8|8h) *- *(10|10h).*$", case=False)] = 10
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(9|9h) *- *(11|11h).*$", case=False)] = 11
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(10|210h) *- *(12|12h).*$", case=False)] = 12

            #Special ones I Found
            huoneiden_lkm[data["Huoneisto"].str.match("^.*(4|4h) *- *(7|7h).*$", case=False)] = 7

            huoneiden_lkm[data["Huoneisto"].str.match("^.*3\(-4\)h.*$", case=False)] = 4
            huoneiden_lkm[data["Huoneisto"].str.match("^.*4\(-5\)h.*$", case=False)] = 5

            huoneiden_lkm[data["Huoneisto"].str.match("^.*3 *\(4\).*$", case=False)] = 4
            huoneiden_lkm[data["Huoneisto"].str.match("^.*4 *\(5\).*$", case=False)] = 5
            huoneiden_lkm[data["Huoneisto"].str.match("^.*5 *\(6\).*$", case=False)] = 6

            # Those who have living rooms seperately
            for i in range(15):
                regex="^.*%i *h.*$"%i

                # eg. huoneisto has 3h or 3 h anywhere AND huoneiden_lkm hasn't been assigned AND it has "oh" somewhere
                huoneiden_lkm[np.logical_and(
                    data["Huoneisto"].str.match(regex, case=False), 
                    np.logical_and(huoneiden_lkm == 0, data["Huoneisto"].str.match("^.*oh.*$", case=False)
                ))] = i+1

            # With mh mark
            for i in range(15):
                regex="^.*%i *mh.*$"%i

                # eg. huoneisto has 3h or 3 h anywhere AND huoneiden_lkm hasn't been assigned AND it has "oh" somewhere
                huoneiden_lkm[np.logical_and(
                    data["Huoneisto"].str.match(regex, case=False), 
                    np.logical_and(huoneiden_lkm == 0, data["Huoneisto"].str.match("^.*oh.*$", case=False)
                ))] = i+1

            # Non livingrooms
            for i in range(15):
                regex="^.*%i *h.*$"%i

                # eg. huoneisto has 3h or 3 h anywhere AND huoneiden_lkm hasn't been assigned AND it does NOT have "oh" somewhere
                huoneiden_lkm[np.logical_and(
                    data["Huoneisto"].str.match(regex, case=False), 
                    np.logical_and(huoneiden_lkm == 0, ~data["Huoneisto"].str.match("^.*oh.*$", case=False)
                ))] = i

                # Since there are only "Neljä huonetta tai enemmän" left to process, so those with [0-3]mh are probabply 4
            for i in range(4):
                regex="^.*%i *mh.*$"%i


                huoneiden_lkm[np.logical_and(
                    data["Huoneisto"].str.match(regex, case=False), 
                    np.logical_and(huoneiden_lkm == 0, ~data["Huoneisto"].str.match("^.*oh.*$", case=False)
                ))] = i

            # More sepcial ones that I found
            huoneiden_lkm[data["Huoneisto"].str.match("4,.*$", case=False)] = 4
            huoneiden_lkm[data["Huoneisto"].str.match("4\+.*$", case=False)] = 4

            return pd.Series(huoneiden_lkm, index = data.index)


        def parsija(text):
            text = str(text)
            #text = text.replace(" ", "") # Poistaa whitespacet kaikkialta
            text = ','.join(text.split(',')[1:]) # Poistaa tavaran ensimmäistä pilkkua
            text = text.strip() # poistaa alun ja lopun whitespacet
            #text = text.replace(",", " ")
            text = re.sub('[^A-Za-z0-9]+', ',', text) # poistaa erikoismerkit ja korvaa ne pilkulla
            text = text.split(",")  #splitataan pilkkujen mukaan

            temp = []
            for sana in text:
                 temp.append(re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", sana).strip())   # lisätään välilyönti numeroiden ja sanojen väliin

            text = (temp)

            text = list(filter(None, text)) # Poistetaan tyhäjt stringit listasta

            text = pd.DataFrame(text)
            dct = {
                   '^tup.*' : 'tupakeittiö',
                   '^avok*' : 'avokeittiö',
                   '^al.*' : 'alkovi', 
                   r'(?i)\b[s]\b' : 'sauna',
                   '^vh.*' : 'vierashuone',
                   r'(?i)\b[k]\b' : 'keittiö',
                   '^khh.*' : 'kodinoitohuone',
                   '^kh.*' : 'kylpyhuone',
                   '^kph.*' : 'kylpyhuone',
                   '^kp.*' : 'kylpyhuone',
                   '^kk.*' : 'keittokomero',
                   '^apuk.*' : 'apukeittiö',
                   '^var*' : 'varasto',
                   '^mh*' : 'makuuhuone',           
                   '^oh*' : 'olohuone',
                   r'(?i)\b[kt]\b' : 'keittiö',
                   '^parve*' : 'parveke',
                   r'(?i)\b[p]\b' : 'parveke',
                   r'(?i)\b[pa]\b' : 'parveke',
                   r'(?i)\b[par]\b' : 'parveke',
                   '^las*' : 'lasitettu parveke',
                   '^pesuh*' : 'pesuhuone',
                   '^psh*' : 'pesuhuone',
                   r'(?i)\b[psh]\b' : 'pesuhuone',
                   r'(?i)\b[kt.]\b' : 'keittiö',
                   r'(?i)\b[kt ]\b' : 'keittiö',
                   r'(?i)\b[th]\b' : 'työhuone',


                  }

            text.replace(dct, regex = True, inplace = True)

            testattava = 0
            sanat = 'sauna'
            if sanat in str(text).lower():    
                testattava = 1



            return testattava 

        #df.Huoneisto.apply(parsija)
        
        def kerros_split(kerros):
    
            try:
                kerros_data = kerros.split('/')
                alempi = int(kerros_data[0])
                ylempi = int(kerros_data[1])
            except:
                alempi, ylempi = 0, 0

            if alempi > ylempi:
                alempi, ylempi = 0, 0
            if alempi < 0:
                alempi, ylempi = 0, 0
            if ylempi < 0:
                alempi, ylempi = 0, 0
            return alempi

        def kerros_split_max(kerros):

            try:
                kerros_data = kerros.split('/')
                alempi = int(kerros_data[0])
                ylempi = int(kerros_data[1])
            except:
                alempi, ylempi = 0, 0

            if alempi > ylempi:
                alempi, ylempi = 0, 0
            if alempi < 0:
                alempi, ylempi = 0, 0
            if ylempi < 0:
                alempi, ylempi = 0, 0
            return ylempi
        
        def imputer(df):
    
            #df = drop_data(df)

            # apufunktio imputoimiseen, etsii yleisimmän entryn listasta
            def most_frequent(List):
                try:
                    return max(set(List), key = List.count)
                except:
                    return str("NAN")


            # Energialuokka

            def e_imputer(e_class):
                e_class = str(e_class)
                if(e_class[0] == 'A'):
                    return 1
                if(e_class[0] == 'B'):
                    return 2
                if(e_class[0] == 'C'):
                    return 3
                if(e_class[0] == 'D'):
                    return 4
                if(e_class[0] == 'E'):
                    return 5
                if(e_class[0] == 'F'):
                    return 6

            Energial = df["Energialuokka"].apply(e_imputer).fillna(7)


            # Kunto

                # Oletetaan, että kunnon puuttuminen on indikaattori kunnosta, eli NAN fillaaminen voisi olla ok päätös. Muuttuja ei korreloinut muiden muuttuhien kanssa.
                #Kunto = df["Kunto"].fillna("NAN")

            mask = df["Kunto"].isna() # muodostetaan maski puuttuvista energialuokituksista
            puuttuvien_vuodet = list(df["Rakennusvuosi"][mask]) # käytetään maskia lukemaan rakennusvuodet puuttuvilta energialuokitus kohteilta
            def kunto_imputer(kunto_class):
                if pd.isna(kunto_class) == 1:
                    if(puuttuvien_vuodet.pop(0) > 2017): # Jos asunnon kunto puuttuu, imputoidaan se hyväksi, jos talo on uudempi kuin 2017, muuten tuntematon kunto
                        return "hyvä"
                    else:
                        return "tyyd."
                else:
                    return kunto_class # Jos ei puutu, OK    

            def Kunto_num(kunto):
                if kunto == 'hyvä':
                    return 2
                if kunto == 'tyyd.':
                    return 1
                if kunto == 'huono':
                    return 0

            Kunto = df["Kunto"].apply(kunto_imputer).apply(Kunto_num)

            # Tontti

                # Fillataan tontti talotyypin tonttien mediaanilla
            mask = df["Tontti"].isna()
            puuttuvien_talot = list(df["Talotyyppi"][mask])    
            def t_imputer(t_class):
                if pd.isna(t_class) == 1:
                    return most_frequent(list(df["Tontti"][df.Talotyyppi == puuttuvien_talot.pop(0)].dropna()))
                else:
                    return t_class

            def Tontti_bin(tontti):
                if tontti == 'oma':
                    return 1
                if tontti == 'vuokra':
                    return 0

            Tontti = df["Tontti"].apply(t_imputer).apply(Tontti_bin)

            # Huoneisto

            Huoneisto = df["Huoneisto"].apply(parsija) # testaa saunalle

            Huoneisto = Huoneisto.fillna("Puuttuva huoneisto")
            Huone_lkm = huoneisto_puhdistaja(df)

            # Kerrokset

            Kerros = df["Kerros"].apply(kerros_split)
            Kerros_max = pd.DataFrame(df["Kerros"].apply(kerros_split_max))

            # Hissi

            def Hissi_bin(text):
                if text == 'on':
                    return 1
                if text == 'ei':
                    return 0

            Hissi = df["Hissi"].apply(Hissi_bin)

            # Postinumero

            Postinumero = df["Postinumero"].fillna(np.mean(df.Postinumero))

            # Rakennusvuosi

            Rakennusvuosi = df["Rakennusvuosi"].apply(int)

            # m2

            m2 = df["m2"]

            # Talotyyppi

            Talotyyppi = df["Talotyyppi"]

            to_name = pd.concat([Talotyyppi, Postinumero, Huoneisto, m2,  Rakennusvuosi, Kerros, Kerros_max, Hissi, Kunto, Tontti, Energial, Huone_lkm]
                              # df["Huoneistotyyppi"]
                              , axis = 1)

            to_name.columns = ["Talotyyppi", "Postinumero", "Sauna", "m2",  "Rakennusvuosi", "Kerros", "Kerros_max", "Hissi", "Kunto", "Tontti", "Energial", "Huone_lkm"]

            return to_name

    

        return imputer(X_)
