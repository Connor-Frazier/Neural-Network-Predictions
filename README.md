# MLFall2019
### Authors: Shagun Bhardwaj, Camil Blanchet and Connor Frazier

### Summary
The goal of this project was to practice creating neural networks using Tensorflow and Keras and to analyze the performance of different models when predicting business fundamental values of public companies. The models used were variations of LTSM and GRU based models with varying amounts of layers and hidden units that all took quarterly business fundamental values as inputs and predicted a vector of four(deemed impotan by research) of those values for the next quarter. These predictions were time series based with the each label being a future vector of values and the data points being reletive previous values from the same company.

### My Role
My role in this project was a majoirty of the engineering and infrastructure work. My responsibilties inlcuded creating the data flow from gathering to cleaning, passing the data to the models for training, training the models, testing the models, and gathering the results. I assisted in domain research and model choice research as well.

#### Run Instrunctions
pip install tensorflow
pip install keras
pip install quandl

The entire testing of all models across all company datasets is run by one script "full-test.py"

run python full-test.py

Then look in your current directory to find the sse_results.csv
