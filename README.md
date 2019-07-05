# Recommendations for the SciStarter Dataset

In this repo we have build a recommender system that uses the content of SciStarter projects in order to recommend users new projects that are most similar to projects they have already done.


## Datasets

There are two main aspects to these datasets.

- Projects: The content of projects were retrieved from the SciStarter API as JSON objects and then converted to the ```data/raw/project_data``` dataframe
- Participation: The records of users interacting with a project (storeed at ```data/raw/participation_data```) 

## Data Exploration

You can see our initial data exploration in a number of notebooks 
- ```notebooks/Cluster SciStarter Projects``` is where we do some initial exploration of the SciStarter projects and build a basic clustering model of the projects
- ```notebooks/Explore Participations``` is where we explore some of the trends in the participations dataset

## Model Training

Our model is split into two parts. 

- First we encode the projects descriptions (which are blocks of plain text) into TF-IDF vectors. 
- Next we train an autoencoder to reduce the dimensionality of this TF-IDF vector to be a much more manageable size.

In order to train the autoencoder you can run through the ```notebooks/Train Autoencoder to Reduce the Size of TF-IDF Vectors``` notebook (the src code is provided in the ```src/``` folder)

Once you have trained the autoencoder you can make recommendations using ```notebooks/Recommend Projects to Users```     
          

### Training Results 

During training our autoencoder we have managed to reduce the dimensionality of our TF-IDF vectors from ~5000 to 10 with an average mean squared loss of 0.006. 

We then measure the success of our recommender using precision and recall. The results for this are very low due to the sparsity of the data (only 0.003% of the potential user/project combinations have actually taken place) and this will be the next challenge to overcome.
    
    
## Author 

- **Thomas Cartwright**: (MSc) Artificial Intelligence, University of Edinburgh
