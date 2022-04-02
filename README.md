# Modelowanie ruchu w sieciach TIN

Instrukcja:
1. Należy wygenerować ruch generatorem, użyto czasu 20000 i czasu awarii 16000
2. Folder traffic należy skopiować do folderu z plikami Pythona
3. Uruchomić plik load_and_save_data.py, zostaną wygenerowane pliki X.npy i Y.npy
4. Uruchomić plik predict.py

W przypadku zmian parametrów, z którymi generowane są dane, należy dostosować zmienną failure_time w pliku predict.py.

Obecnie dane są predykowane dla pary węzłów (5, 8). Dla innej pary należy dostosować zmienne predict_from i predict_to w pliku load_and_save_data.py
