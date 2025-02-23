# How to run

In this instruction, I use pyenv for the virtual environment and python 3.9.17. It actually works on python >= 3.9

- pip install -r requirements.txt

# Download Pytesseract

         download Pytesseract from here
        
        [Link] (https://github.com/UB-Mannheim/tesseract/wiki)

        after that add the Pytesseract to your system enviroment variabel 

                1. Search windows and type : Edit the system enviroment variabels
                2. Click Enviroment Variabels
                3. Go to enviroment variabels seearch path and click edit
                4. Click new and add to your path Example :(C:\User\nabil\Program Files\Tesseract-OCR)


## Prepare Environment
 Create new File .env and add your Api Key 

        Example : 
                
                GOOGLE_API_KEY=(Your_API_KEy)
                # Do not use the tuple to your API key 



# Run the app

        streamlit run app.py


