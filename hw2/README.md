# SDML Homework 2

## Task 1

* Given: the food consumption history of each user
* Predict: what kind of ‘new’ food each user will consume in the future
* Historical consumption data for each user can be found in rating_train.csv
* Since it is a consumption diary, users may eat repeated food items

### Top-1 tips

* Warp-loss
* Variational Autoencoders for Collaborative Filtering

## References

[ALS Implicit Collaborative Filtering](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe)

## Packages

[Implicit](https://implicit.readthedocs.io/en/latest/)

[Surprise](https://surprise.readthedocs.io/en/stable/index.html)

### Implicit

#### Loading Data

```python
# read in triples of user/artist/playcount from the input dataset
data = pandas.read_table("usersha1-artmbid-artname-plays.tsv",
                         usecols=[0, 2, 3],
                         names=['user', 'artist', 'plays'])

# map each artist and user to a unique numeric value
data['user'] = data['user'].astype("category")
data['artist'] = data['artist'].astype("category")

# create a sparse matrix of all the artist/user/play triples
plays = coo_matrix((data['plays'].astype(float),
                   (data['artist'].cat.codes,
                    data['user'].cat.codes)))
```
