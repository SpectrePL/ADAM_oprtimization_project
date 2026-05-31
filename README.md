# ADAM_oprtimization_project

W pliku requriments znajdują się wykorzystywame biblioteki i ich wersje

Instrukcja obsługi kodu:
1. W pliku eksperymenty.xml znajdują się konfiguracje poszczególnych uruchomień
2. W celu sprawdzenia nowego przypadku należy:
 a) (zalecane) Stworzyć nowy eksperyment na bazie formatu pozostałych, dodać go do pliku
 b) Stworzyć nowy plik xml z pojedynczym ekspetymentem i w linii 17 zmienić adres pliku na nowy (należy pamiętać by umieścić na samej górze podstawowe ustawienia parametrów algorytmu ADAM)
4. Następnie należy w folderze pliki odpalić plik main
5. Wygenerowane wyniki (wykres 3D, poziomicowy i raporty txt) znajdują się w folderze wyniki

Template eksperymentu:
```xml
<experiment id="Numer ID (np EXP001)"> (W przypadku chęci stworzenia kilku trajektorii na jednym wykresie należy stworzyć drugi eksperyment i nadać mu identyczne id)
            <name>Nazwa eksperymentu</name>
            <function_name>Nazwa funkcji (klasy z pliku funkcje.py)</function_name>
            <start_point>
                <coordinate>Wartość x1</coordinate>
                <coordinate>Wartość x2</coordinate>
            </start_point>
            <is_constrained>true/false</is_constrained>
            
            <penalty_config>(jeśli is_constrained = false, kontener powinien być pusty)
                <r>wartość r(float) - stopień wpływu kary</r>
                <constraints>
					(jeśli chce się zastosować wiele funkcji kary należy wstawić kilka kontenertów typu constraint)
                    <constraint type="equality/inequality"> (typ ograniczenia) 
                        <class_name>Nazwa funkcji kary (z pliku functions.py) </class_name>
                        <parameters>(parametry funkcji kary, zgodne z parametrami opisanymi w pliku functions.py)
                            <param name="a">float</param>
                            <param name="b">float</param>
                            <param name="c">float</param>
                        </parameters>
                    </constraint>
                </constraints>
            </penalty_config>

			<hyperparameters>(kontener całkowicie opcjonalny; każdy z parametrów również opcjonalny (nie ustawiony parametr brany jest z ustawień domyślnych))
                <param name="alpha">float</param>
				<param name="beta1">float</param>
        		<param name="beta2">float</param>
        		<param name="eps">float</param>
                <param name="max_iter">int</param>
            </hyperparameters>
        </experiment>
