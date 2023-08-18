import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import csv
import numpy as np
import matplotlib.pyplot as plt
from GHEtool import Borefield, FluidData, GroundData, PipeData
import pygfunction as gt
import math
import itertools
from lin_reg import *


st.set_page_config(page_title="O store COP-beregning", page_icon="🔥")

with open("styles/main.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


## ------------------------------------------------------------------------------------------------------##

@st.cache_data
def plot_datablad(valg_av_vp):
    if valg_av_vp == 'Mitsubishi CRHV-P600YA-HPB':
        # COP-Verdier fra datablad
        databladtemp35 = np.array([-5,-2,0,2,5,10,15])
        COP_data35 = np.array([3.68, 4.03, 4.23, 4.41, 4.56, 5.04, 5.42])
        databladtemp45 = np.array([-2,0,2,5,10,15])
        COP_data45 = np.array([3.3, 3.47, 3.61, 3.77, 4.11, 4.4])
    else:
        pass

    #Kjører lineær regresjon på COP-verdier fra datablad:
    
    lin_COP_data35 = lin_reg(databladtemp35,COP_data35)
    lin_COP_data45 = lin_reg(databladtemp45,COP_data45)

    # Plotter COP-verdier fra datablad

    fig = plt.figure()
    plt.plot(databladtemp35,COP_data35,'o', color='#1d3c34')
    plt.plot(databladtemp45,COP_data45,'o', color='#48a23f')
    plt.plot(databladtemp35,lin_COP_data35, color='#1d3c34')
    plt.plot(databladtemp45,lin_COP_data45, color='#48a23f')
    plt.legend(['Datablad 35 \u2103', 'Datablad 45 \u2103','Lineær 35 \u2103','Lineær 45 \u2103'])
    plt.xlabel('Turtemperatur fra brønn (\u2103)')
    plt.ylabel('COP')
    plt.title('COP-verdier fra datablad, ved 100 % kapasitet', fontsize = 20)
    plt.grid(True)
    st.pyplot(fig)

    # Parametre i uttrykket for den lineære regresjonen av COP fra datablad
    stigtall35 = (lin_COP_data35[-1]-lin_COP_data35[0])/(databladtemp35[-1]-databladtemp35[0])
    konstledd35 = lin_COP_data35[-1]-stigtall35*databladtemp35[-1]

    stigtall45 = (lin_COP_data45[-1]-lin_COP_data45[0])/(databladtemp45[-1]-databladtemp45[0])
    konstledd45 = lin_COP_data45[-1]-stigtall45*databladtemp45[-1]

    return stigtall35,konstledd35,stigtall45,konstledd45


# Funksjon for Lineær interpolering:
def lin_interp(x,x1,x2,y1,y2):
    y = y1+(x-x1)*(y2-y1)/(x2-x1)
    return y

def bronnlast_fra_COP(grunnlast,cop,virkgrad):
    ellast = grunnlast/cop*virkgrad
    bronnlast = grunnlast-ellast

    return bronnlast


def bestem_turtemp(utetemp,utetemp_for_maks_turtemp,utetemp_for_min_turtemp,maks_turtemp,min_turtemp):
    turtemp = np.zeros(len(utetemp))
    for i in range(0,len(utetemp)):
        if utetemp[i]<utetemp_for_maks_turtemp:
            turtemp[i] = maks_turtemp
        elif utetemp[i]>utetemp_for_min_turtemp:
            turtemp[i] = min_turtemp
        else:
            #Lineær interpolering:
            turtemp[i] = lin_interp(utetemp[i],utetemp_for_maks_turtemp,utetemp_for_min_turtemp,maks_turtemp,min_turtemp)
    return turtemp


def finn_ny_COP(bronntemp_vektor,stigtall35,konstledd35,stigtall45,konstledd45,turtemp,maks_turtemp,min_turtemp):
    # COP som funksjon av turtemp (basert på COP som funksjon av brønntemp)
    nyCOP = np.zeros(len(turtemp))
    for i in range(0,len(turtemp)):
        if turtemp[i] == maks_turtemp:
            nyCOP[i] = stigtall45*bronntemp_vektor[i]+konstledd45
        elif turtemp[i] == min_turtemp:
            nyCOP[i] = stigtall35*bronntemp_vektor[i]+konstledd35
        else:
            stigtall_interp = lin_interp(turtemp[i],min_turtemp,maks_turtemp,stigtall35,stigtall45)
            konstledd_interp = lin_interp(turtemp[i],min_turtemp,maks_turtemp,konstledd35,konstledd45)
            COP_interp = stigtall_interp*bronntemp_vektor[i]+konstledd_interp
            nyCOP[i] = COP_interp
    nyCOP=np.array(nyCOP)

    COP=nyCOP
    return COP


## ------------------------------------------------------------------------------------------------------##

class O_store_COP_beregning:
    def __init__(self):
        pass
    
    def kjor_hele(self):
        self.streamlit_input()
        with st.spinner("Grubler ... 🤔"):
            if self.kjor_knapp == True:
                self.last_inn_varmebehov()
                self.grunnlast_fra_varmelast()
                self.dybde_COP_sloyfe()
                self.skriv_ut_resultater()


    def streamlit_input(self): 

        st.title('O store COP-beregning 🤯')
        st.markdown('Laget av Åsmund Fossum 👨🏼‍💻')
        st.markdown('---')

        st.subheader('Data om grunnen')
        c1, c2 = st.columns(2)
        with c1:
            self.LEDNINGSEVNE = st.number_input('Termisk ledningsevne til grunnen (W/mK)',value=3.5, max_value=float(10), min_value=0.1, step=0.1)
        with c2:
            self.UFORST_TEMP = st.number_input('Uforstyrret temperatur (\u2103)',value=7.5, step=0.5)
        st.markdown('---')


        st.subheader('Brønnkonfigurasjon')
        self.MIN_BRONNTEMP = st.number_input('Laveste tillatte gjennomsnittlige kollektorvæsketemperatur etter 25 år (\u2103)', value = float(0), step=0.1)
        
        c1,c2 = st.columns(2)
        with c1:
            self.MAKS_DYBDE = st.number_input('Maksimal tillatte dybde per brønn (m)', value=300, step=10)
        with c2:
            self.DIAM = st.number_input('Diameter til brønnen (mm)', value=115, min_value=100, max_value = 150, step=1)

        self.onsket_konfig = st.selectbox(label='Foretrukket brønnkonfigurasjon', options=['Linje','"L"-formet','Kvadrat (tilnærmet)','Rektangel med fastsatt side','Boks','Boks med fastsatt side'], index=0)
        
        c1, c2 = st.columns(2)
        with c1:
            self.avstand = st.number_input('Avstand mellom nærliggende brønner (m)', value=15, min_value=1, step=1)
        with c2: 
            if self.onsket_konfig == 'Rektangel med fastsatt side' or self.onsket_konfig == 'Boks med fastsatt side':
                self.fastsatt_side = st.number_input('Antall brønner langs en side', value=2, min_value=2, step=1)
        st.markdown('---')

        st.subheader('Valg av varmepumpe')
        c1, c2, = st.columns(2)
        with c1:
            self.type_VP = st.selectbox(label='Velg varmepumpe',options=['Mitsubishi CRHV-P600YA-HPB', 'VP 2', 'VP 3', 'VP 4'])
        with c2:
            self.VIRKGRAD = st.number_input('Virkningsgrad til kompressor (%)',value=80, max_value=100, min_value=10, step=1)/100

       
        [self.stigtall35,self.konstledd35,self.stigtall45,self.konstledd45] = plot_datablad(self.type_VP)
        self.MAKS_TURTEMP = 45
        self.MIN_TURTEMP = 35
    
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Maksimal turtemperatur", f"{self.MAKS_TURTEMP} \u2103")
        with c2:
            st.metric("Minimal turtemperatur", f"{self.MIN_TURTEMP} \u2103")
        
        c1,c2 = st.columns(2)
        with c1:
            self.UTETEMP_FOR_MAKS_TURTEMP = st.number_input('Utetemperatur som gir denne turtemperaturen*', value=-15)
        with c2:
            self.UTETEMP_FOR_MIN_TURTEMP = st.number_input('Utetemperatur som gir denne turtemperaturen*', value=15)
        st.markdown('* \* Det antas her at varmepumpen kjører med konstant kapasitet hele tiden, slik at COP kun avhenger av brønntemperatur og turtemperatur (sistnevnte avhenger videre utelufttemperatur). Dette innebærer at kompressoren er turtallsregulert. I mange anlegg vil COP også avhenge av kapasiteten, som igjen bestemmes av energibehovet.')
        
        heat_carrier_fluid_types = ["HX24", "HX35", "Kilfrost GEO 24%", "Kilfrost GEO 32%", "Kilfrost GEO 35%"]
        heat_carrier_fluid_densities = [970.5, 955, 1105.5, 1136.2, 1150.6]
        heat_carrier_fluid_capacities = [4.298, 4.061, 3.455, 3.251, 3.156]

        c1,c2 = st.columns(2)
        with c1:
            self.TYPE_KOLL_VAESKE = st.selectbox('Type kollektorvæske i varmepumpen', options=heat_carrier_fluid_types, index=0)
        with c2:
            self.VOLSTROM_KOLLVAESKE = st.number_input('Volumstrøm til kollektorvæsken (l/s/brønn)', value=0.5, min_value=0.1, step=0.1)
        
        for i in range(0,len(heat_carrier_fluid_types)):
            if self.TYPE_KOLL_VAESKE == heat_carrier_fluid_types[i]:
                self.fluid_tetthet = heat_carrier_fluid_densities[i]
                self.fluid_kapasitet = heat_carrier_fluid_capacities[i]
    
        st.markdown('---')

        st.subheader('Varmebehov')
        #self.varmelast_fil = st.file_uploader('CSV-fil med timesoppløst varmeenergibehov for et år',type='csv')
        self.varmelast_fil = st.file_uploader('Last opp Excel-fil med to kolonner som inneholder hhv. varmeenergibehov og utelufttemperatur, begge med timesoppløsning over ett år.',type='xlsx')
        self.DEKGRAD = st.number_input('Ønsket dekningsgrad (%)',value=90, max_value=100, min_value=1, step=1)/100
        
        st.markdown('')
        c1,c2,c3,c4,c5,c6,c7= st.columns(7)
        with c4:
            self.kjor_knapp = st.button(' Kjør 🚂 ')

        st.markdown('---')


        self.ANTALL_AAR = 25
        self.COP = 3.5                       # Resultat uavhengig av denne
        self.DYBDE_STARTGJETT = 250          # Resultat uavhengig av denne
        self.TERM_MOTSTAND = 0.08

    def last_inn_varmebehov(self): 
        df = pd.read_excel(self.varmelast_fil)
        df = df.to_numpy()
        df = np.swapaxes(df,0,1)
        self.varmelast = df[0,:]
        self.utetemp = df[1,:]

    def grunnlast_fra_varmelast(self):
        @st.cache_data
        def grunnlast_fra_varmelast(varmelast,DEKGRAD,ANTALL_AAR):
            grunnlast = 1*varmelast
            spisslast = 1*varmelast

            maks = np.max(varmelast)
            mini = np.min(varmelast)

            for kap in np.arange(0.8*maks,mini,-0.01*maks):  #Sjekker fra 80% av makslast og nedover med steglengde 1% av denne.
                for i in range(0, len(grunnlast)):
                    if grunnlast[i]>=kap:
                        grunnlast[i]=kap
                    else:
                        grunnlast[i]=grunnlast[i]
                    
                for j in range(0,len(spisslast)):
                        if grunnlast[j]>=kap:
                            spisslast[j]=varmelast[j]-kap
                        else:
                            spisslast[j]=0

                if np.sum(grunnlast)/(np.sum(varmelast))<DEKGRAD:
                    break

            grunnlast = np.array(grunnlast)
            GRUNNLAST = np.hstack(ANTALL_AAR*[grunnlast])
            return GRUNNLAST

        self.GRUNNLAST = grunnlast_fra_varmelast(self.varmelast,self.DEKGRAD,self.ANTALL_AAR)


    def GHE_tool_bronndybde(self,antall_bronner1,antall_bronner2,bronnlast,dybde_GHE):
        data = GroundData(self.LEDNINGSEVNE, self.UFORST_TEMP, self.TERM_MOTSTAND, 2.518 * 10**6)    # Siste parameter: Volumetric heat capacity of ground
        
        if self.onsket_konfig == '"L"-formet':
            bronnfelt = gt.boreholes.L_shaped_field(N_1=antall_bronner1, N_2=antall_bronner2, B_1=self.avstand, B_2=self.avstand, H=dybde_GHE, D = 10, r_b = float(self.DIAM)/2000)
            self.tot_ant_bronner = antall_bronner1+antall_bronner2-1
        
        elif self.onsket_konfig == 'Boks' or self.onsket_konfig == 'Boks med fastsatt side':
            bronnfelt = gt.boreholes.box_shaped_field(N_1=antall_bronner1, N_2=antall_bronner2, B_1=self.avstand, B_2=self.avstand, H=dybde_GHE, D = 10, r_b = float(self.DIAM)/2000)
            if antall_bronner1>=3 and antall_bronner2>=2:
                self.tot_ant_bronner = 2*antall_bronner2+2*(antall_bronner1-2)
            else:
                self.tot_ant_bronner = antall_bronner1*antall_bronner2
        
        else:
            bronnfelt = gt.boreholes.rectangle_field(N_1=antall_bronner1, N_2=antall_bronner2, B_1=self.avstand, B_2=self.avstand, H=dybde_GHE, D = 10, r_b = float(self.DIAM)/2000) # Siste to parametre: Boreholde buried depth og borehole radius (m)
            self.tot_ant_bronner = antall_bronner1*antall_bronner2
        
        borefield_gt = bronnfelt

        borefield = Borefield(simulation_period=self.ANTALL_AAR)
        borefield.set_ground_parameters(data)
        borefield.set_borefield(borefield_gt)        
        #borefield.set_hourly_heating_load(bronnlast)

        borefield.hourly_heating_load = bronnlast[-8760:]
        borefield.hourly_cooling_load = np.zeros(8760)

        borefield.set_max_ground_temperature(16)   # maximum temperature   Utgjør ingen forskjell å endre på denne.
        borefield.set_min_ground_temperature(self.MIN_BRONNTEMP)    # minimum temperature
        
        self.dybde_GHE = borefield.size(dybde_GHE, L4_sizing=True)
        self.bronntemp_vegg = borefield.Tb
        
        snitt_koll_vaeske_temp = borefield.results_peak_heating

        Q = (bronnlast)/self.tot_ant_bronner
        delta_T = (Q*1000)/(self.fluid_tetthet*self.VOLSTROM_KOLLVAESKE*self.fluid_kapasitet)

        self.bronntemp_tur = snitt_koll_vaeske_temp + delta_T/2
        self.bronntemp_retur = snitt_koll_vaeske_temp - delta_T/2   

    def dybde_COP_sloyfe(self):
        
        turtemp = bestem_turtemp(self.utetemp,self.UTETEMP_FOR_MAKS_TURTEMP,self.UTETEMP_FOR_MIN_TURTEMP,self.MAKS_TURTEMP,self.MIN_TURTEMP)
        TURTEMP = np.hstack(self.ANTALL_AAR*[turtemp])

        COP = np.array([self.COP]*8760*self.ANTALL_AAR)
        self.dybde_GHE = self.DYBDE_STARTGJETT
        ant_bronner1 = 1
        
        if self.onsket_konfig == 'Rektangel med fastsatt side' or self.onsket_konfig == 'Boks med fastsatt side':
            ant_bronner2 = self.fastsatt_side
        else:
            ant_bronner2 = 1

        # For-løkke --------------------------------------------------------------------------------------------------------------------------------------------
        for k in range(0,20):

            BRONNLAST = bronnlast_fra_COP(self.GRUNNLAST,COP,self.VIRKGRAD)

            self.GHE_tool_bronndybde(ant_bronner1,ant_bronner2,BRONNLAST,self.dybde_GHE)

            if k==0 and self.dybde_GHE >= self.MAKS_DYBDE:
                ant_bronner1 = math.ceil(self.dybde_GHE/self.MAKS_DYBDE) #runder alltid opp
                print(ant_bronner1)

            print('DYBDE:',self.dybde_GHE)

            if k > 0 and self.dybde_GHE >= self.MAKS_DYBDE:
                ant_bronner1 = ant_bronner1+1
            
            print(ant_bronner1)
            print(ant_bronner2)
            
            nyCOP = finn_ny_COP(self.bronntemp_tur,self.stigtall35,self.konstledd35,self.stigtall45,self.konstledd45,TURTEMP,self.MAKS_TURTEMP,self.MIN_TURTEMP)
            
            if np.mean(np.abs(nyCOP-COP))<0.1 and self.dybde_GHE <= self.MAKS_DYBDE: 
                COP = nyCOP
                if self.onsket_konfig == 'Kvadrat (tilnærmet)' and ant_bronner2 == 1 and ant_bronner1 != 1:
                    ant_per_side = math.ceil(np.sqrt(ant_bronner1))
                    ant_bronner1 = ant_per_side-1
                    ant_bronner2 = ant_per_side

                elif self.onsket_konfig == 'Boks' and ant_bronner2 == 1 and ant_bronner1 >= 3:
                    if ant_bronner1 == 3 or ant_bronner1 == 4:
                        ant_bronner1 = 2
                        ant_bronner2 = 2
                    elif ant_bronner1 == 5 or ant_bronner1 == 6:
                        ant_bronner1 = 3
                        ant_bronner2 = 2
                    else:
                        ant_per_side = math.ceil(np.sqrt(ant_bronner1))
                        ant_bronner1 = ant_per_side
                        ant_bronner2 = ant_per_side

                elif self.onsket_konfig == '"L"-formet' and ant_bronner2 == 1 and ant_bronner1 >= 3:
                    ant_per_side = math.ceil(ant_bronner1/2)
                    if (ant_bronner1 % 2) == 0: #partall
                        ant_bronner2 = ant_per_side+1
                    else: #oddetall
                        ant_bronner2 = ant_per_side
                    ant_bronner1 = ant_per_side
                
                elif self.onsket_konfig == 'Rektangel med fastsatt side' or self.onsket_konfig == 'Boks med fastsatt side':
                    break
                elif self.onsket_konfig == 'Sirkel':
                    break
                else:  # Hvis ønkset konfig. er linje
                    break
            else:
                COP = nyCOP
            print('Antall iterasjoner:', k+1)
        # -------------------------------------------------------------------------------------------------------------------------------------------

        self.ant_bronner1 = ant_bronner1
        self.ant_bronner2 = ant_bronner2
        self.COP = COP

    def skriv_ut_resultater(self):

        st.header('Resultater')
        st.subheader('Foreslått brønnkonfigurasjon')
        c1, c2, c3 = st.columns(3)
        with c1:
            if self.onsket_konfig == '"L"-formet':
                st.metric("Brønnkonfigurasjon", f"{self.ant_bronner1} x {self.ant_bronner2} (L)")
            elif self.onsket_konfig == 'Boks' or self.onsket_konfig == 'Boks med fastsatt side':
                st.metric("Brønnkonfigurasjon", f"{self.ant_bronner1} x {self.ant_bronner2} (Boks)")
            else:
                st.metric("Brønnkonfigurasjon", f"{self.ant_bronner1} x {self.ant_bronner2}")
        with c2:
            st.metric("Totalt antall brønner", f"{self.tot_ant_bronner}")
        with c3:
            st.metric("Dybden til hver brønn", f"{round(self.dybde_GHE,1)} m")
        
        r = self.ant_bronner2
        c = self.ant_bronner1
        x = [i % c * self.avstand for i in range(r*c)]
        y = [i // c * self.avstand for i in range(r*c)]
        fig2, ax2 = plt.subplots()
        if self.onsket_konfig == '"L"-formet':
            ax2.scatter(x,np.zeros(len(x)),color = '#367A2F')
            ax2.scatter(np.zeros(len(y)),y,color = '#367A2F')
        elif self.onsket_konfig == 'Boks' or 'Boks med fastsatt side':
            ax2.scatter(x,np.zeros(len(x)),color = '#367A2F')
            ax2.scatter(np.zeros(len(y)),y,color = '#367A2F')
            ax2.scatter(x,np.ones(len(x))*y[-1],color = '#367A2F')
            ax2.scatter(np.ones(len(y))*x[-1],y,color = '#367A2F')
        else:
            ax2.scatter(x,y,color = '#367A2F')
        x_ticks = [i * self.avstand for i in range(c)]
        y_ticks = [i * self.avstand for i in range(r)]
        ax2.set_xticks(x_ticks)
        ax2.set_yticks(y_ticks)
        ax2.grid(True)
        ax2.set_title('Brønnkonfigurasjon')
        ax2.set_xlabel('Avstand (m)')
        ax2.set_ylabel('Avstand (m)')
        ax2.axis('equal')
        st.pyplot(fig2)

        if self.dybde_GHE <= 0.8*self.MAKS_DYBDE and self.onsket_konfig != 'Linje' and self.onsket_konfig != '"L"-formet' and self.tot_ant_bronner >= 3:
            st.markdown(':red[MERK: Denne konfigurasjonen og det gitte forbruket gir sannsynligvis flere og grunnere brønner enn nødvendig. Prøv eventuelt å endre brønnkonfigurasjon i menyen over til Linje eller "L"-formet.]')

        st.subheader('Brønntemperatur')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Laveste turtemp. (fra brønn)", f"{round(np.min(self.bronntemp_tur),2)} \u2103")
        with c2:
            st.metric("Laveste returtemp. (til brønn)", f"{round(np.min(self.bronntemp_retur),2)} \u2103")
        with c3:
            st.metric("Laveste veggtemp. i brønnen", f"{round(np.min(self.bronntemp_vegg),2)} \u2103")

        fig = plt.figure()
        plt.plot(self.bronntemp_retur,color = '#367A2F')
        plt.plot(self.bronntemp_tur,color = '#1d3c34')
        plt.plot(self.bronntemp_vegg,color = '#FFC358')
        plt.title((f'Brønntemperaturen gjennom {self.ANTALL_AAR} år'), fontsize = 20)
        plt.xlabel('Timer')
        plt.ylabel("Temperatur (\u2103)")
        plt.legend(['Returtemperatur til brønn','Turtemperatur fra brønn','Veggtemperatur i brønn'])
        plt.grid(True)
        st.pyplot(fig)

        st.subheader('COP')
        snittCOP = round(np.mean(self.COP),2)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Maksimal COP", f"{round(np.max(self.COP),2)}")
        with c2:
            st.metric("Minimal COP", f"{round(np.min(self.COP),2)}")
        with c3:
            st.metric("Gjennomsnittlig COP", f"{snittCOP}")

        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure()
            plt.plot(self.COP,color = '#367A2F')
            plt.plot(np.array([snittCOP]*len(self.COP)),color='#FFC358')
            plt.title((f'Variasjoner i COP gjennom {self.ANTALL_AAR} år'), fontsize = 20)
            plt.xlabel('Timer')
            plt.ylabel("COP")
            plt.legend(['COP','Gjennomsnitt'])
            plt.grid(True)
            st.pyplot(fig)
        with c2:
            fig = plt.figure()
            plt.plot(self.COP[-8760:],color = '#367A2F')
            plt.plot(np.array([snittCOP]*8760),color='#FFC358')
            plt.title('Variasjoner i COP gjennom siste år', fontsize = 20)
            plt.xlabel('Timer')
            plt.ylabel("COP")
            plt.legend(['COP','Gjennomsnitt'])
            plt.grid(True)
            st.pyplot(fig)

        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure()
            plt.plot(self.utetemp,self.COP[-8760:],'.',color = '#367A2F')
            plt.title(('COP som funksjon av utetemp. (ett år)'), fontsize = 20)
            plt.xlabel('Utetemp (\u2103)')
            plt.ylabel("COP")
            plt.grid(True)
            #plt.legend(['Returtemperatur til brønn','Brønntemperatur'])
            st.pyplot(fig)
        with c2:
            fig = plt.figure()
            plt.plot(self.bronntemp_tur[-8760:],self.COP[-8760:],'.',color = '#367A2F')
            plt.title(('COP som funksjon av turtemp., siste år'), fontsize = 20)
            plt.xlabel('Turtemp (\u2103)')
            plt.ylabel("COP")
            plt.grid(True)
            #plt.legend(['Returtemperatur til brønn','Brønntemperatur'])
            st.pyplot(fig)

        st.markdown('---')



O_store_COP_beregning().kjor_hele()