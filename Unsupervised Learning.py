##################################Clustering####################################

#####Clusering 2D points#####
#arrays: points and new_points 

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters = 3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)


#####Inspect the Clustering#####
#new_points is an array of points and Labels is the array of their cluster labels

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:, 0]
ys = new_points[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s = 50)
plt.show()


#####How many clusters of grain?#####
#choose a good number of clusters for a dataset using the k-means inertia
#array: samples
#imported for you: KMeans, and PyPlot (plt)

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


#####Evaluating the grain clustering#####
#you have the array samples of grain samples, and a list of varieties
#Pandas (pd) and KMeans have been imported

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters = 3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


#####Scaling data for clustering - standardizing#####
#build a pipeline to standardize and cluster the data

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler - Standardize!: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters = 4)

# Create pipeline - pass them in as arguments: pipeline
pipeline = make_pipeline(scaler, kmeans)

#now use the stadardization and clustering pipeline to cluster the data
#create a cross-tabulation to compare the cluster labels with the fish species

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels -- or you could combine the with .fit_predict: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)


#####Clustering Stocks using KMeans#####
#NumPy array: movements
#KMeans and make_pipeline have been imported 
#some stocks are more expensive than others, to account for this, include a Normalizer
#The Normalizer will separately transform each company's stock to a relative scale before clustering
#StandardScaler() stadardizes features, whereas Normalizer() rescales each sample independently of the other

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters = 10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

#Which companies have stock prices that tend to change in the same way?
#Inspect cluster labels to find out
# list companies is available

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))


#########################################Hierarchical Clustering###################################
#SciPy linkage() performs hierarchical clustering on an array of samples


#####Hierarchical clustering of grain data#####
#Use linkage() to obtain a hierarchical clustering of grain samples
#Use dendrogram() to visualize the result
#A sample of the rain measurements is provided in the array samples
#The variety of each grain sample is given by the list varieties

# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage - linkage() performs hierarchical clustering on an array of samples: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


#####Hierarchies of stocks#####
#NumPy array of price movements 
#List of companies#need to use normalize() from sklearn.preprocessing instead of Normalizer
#linkage and dendrogram have already been imported from sklearn.cluster.hierarchy
#PyPlot has been imported as plt

# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


#####Different linkage, different hierarchical cluster#####
#complete vs single linkage: two farthest points or two closest points
#you are given an array samples
#the list country_names gives the name of each voting country_names

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage - single instead of complete last time: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()


#####Extracting the cluster labels#####
#Use the fcluster() function to extract the cluster labels for this intermediate clustering
#compare the labels with the grain varieties using a cross-tabulation
##hierarchical clustering has already been performed
#mergings is the result of the linkage() function
#The list varieties gives the variety of each grain sample

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


#####t-SNE visualization of grain dataset#####
#you are given an array samples of grain samples and a list variety_numbers giving the variety number of each grain sample
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()


#####a t-SNE map of the stock market#####
#stock price movements are in the array normalized_movements
#the list companies gives the name of each company
#PyPlot (plt) has been imported

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()


#####################################Dimension Reduction##############################
#decorrelate

#####Correlated data in nature#####
#you are given an array grains with the width and length of samples of grain.
#you suspect that width and length will be correlated
#first look at a scatter plot

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)


#####Decorrelating the grain measurements with PCA#####
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


#####The first principle component#####
#The first principal component of the data is the direction in which the data varies the most
#find the first principal component
#The array grains gives the length and width of the grain samples
#PyPlot and PCA have been imported

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()


#####Variance of the PCA features#####
#The fish dataset is 6-dimensional, but what is the intrinsic dimension?
#plot the variances to find out
#The intrinsic dimension is the number of PCA features with significant variance
#samples is a 2D array - this needs to be standardized first

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


#####Dimension reduction of the fish measurements#####
#retain only the most important components
#the fish measurements have already been scaled for you, available as scaled_samples

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)


#####tf-idf word frequency array#####
#use TfidVectorizer from sklearn - it transforms a list of documents into a word frequency array and outputs as a csr_matrix
#it has fit() and transform() functions
#you are given a list documents of toy documents about pets

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)


#####Clustering Wikipedia part I#####
#combine knowledge of Truncated SVD and kmeans to cluster popular wikipedia pages
#in this exercise build the pipeline, in the following apply it to a word-frequency array of some wikipedia articles

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

#use the pipeline to cluster the wiki articles

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))


####################################Non-negative Matrix Factorization##############################
'''expresses samples as combinations of interpretable parts. For example, it expresses documents
as combinations of topics, and images in terms of commonly occurring visual patterns. You'll 
also learn to use NMF to build recommender systems that can find you similar articles to read, 
or musical artists that match your listening history'''

#####NMF applied to wikipedia articles#####
#using the tf-idf word-frequncy array of wiki articles, fit the model and transform the articles

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)

#explore the NMF features 
#the list titles is available each each wiki article
#When investigating the features, notice the NMF feature 3 as the highest value.

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway',:])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington',:])

#when NMF is applied to documents, the components correspond to topics of the documents
#NMF features reconstruct the documents from the topics
# recognise the topic that the articles about Anne and Denzel have in commonly

# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())


#####Explore the LED digits dataset#####
#Use NMF to decompose grayscale images to their commonly occurring patterns
#First explore the image dataset and see how it is encoded as an array
#You are given 100 images as a 2D array samples - each row represents a single 13 x 8 image
#The images in your dataset are pictures of an LED digital display

# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape((13, 8))

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


#####NMF learns the parts of images#####
#You are given the digit images as a 2d array samples
#You are also proved with a function show_As_image() that displays the image encoded by any 1D array

def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
	
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)


#####NOTE: PCA doesn't learn parts#####
#unlike NMF, PCA doesn't learn the parts of things, components do not correspond to topics or parts of an image

#######################################Building Recommender Systems using NMF##############################
#Use NMF features and cosine similarity to find similar articles

#####Which articles are similar to 'Cristiano Ronaldo'?#####
#The NMF features obtained earlier are available, while titles is a list of the article titles

# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products - cosine similarity: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())


#####Recommend musical artists#####
#you are given a sparse array artists whose rows correspond to artists and columns to users - showing number of times listened to by each users
#build a pipeline and transform the array into normalized NMF features.
#The first step in the pipeline is MaxAbsScaler, to transform the data so all users have same influence in the model regardless of how many different artists they have listened to

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


#Suppose you were a fan of Bruce Springsteen - what other artists may you like?
#use the NMF features and the cosine similarity to find similar artists
#norm_features is the array containing the normalized NMF features as rows
#the names of the artists are available as the list artist_names

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())

