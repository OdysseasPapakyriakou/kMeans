Name: Odysseas Papakyriakou

BACKGROUND INFORMATION:

This is a project I created as part of the requirements of a course in Data Processing
at the University of Amsterdam. The aim was to read, transform, analize, and visualize
a dataset of our choice, by defining our own research question. 

INSTRUCTIONS TO RUN THE PROGRAM:

- open the file main.py
- type 'jupyter notebook' in the terminal
- from the pop-up window that appears, open 'Final Notebook.ipynb'
- from the 'Final Notebook' run the first cell and every subsequent cell that contains code
- the visualizations are in the final part of the notebook, 
  and will appear after running these cells

INFORMATION ABOUT THE PROJECT AND THE FILES:

- The project contains several packages, each of which has several functions, 
  so that these functions can be called in the Notebook
- Here are the names of the packages with a very brief description:
	1. data_processing	--> reading the necessary datasets
	2. transformations	--> performs all necessary transformations for the project
	3. transformed_datasets	--> reads the transformed datasets so they can be used later
	4. visualizations	--> creates the visualizations

- The functions of the packages are documented using NumPy style documentation.
- To see the detailed documentation of the packages and the functions they contain,
  type: 'pydoc', the name of the package, followed by '.', 
	followed by 'functions' in the terminal of main.py.
  Example: pydoc visualizations.functions

  If you cannot type in the terminal, this is probably because Jupyter is still running.
  Press ctr+c in the terminal to terminate Jupyter, so that you can see the documentation.
  
  The documentation will show the description of the package along with a detailed 
  description of each function within the package. The description of the functions 
  contains information about what it does, the parameters it accepts, and what it returns.