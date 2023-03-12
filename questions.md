# Questions during modelling 

1. during cross val, some folds have extremely high error (only for a couple models though)
2. what base features should we use for illustration of feature engineering importance?
3. should we only include tuned models in the final table or stick to default params or semi default
4. removal of outliers and general data cleaning. Some locations listed as overseas too. 
5. what is a suitable number of plots to include? do papers have appendix for extra?
    * some very interesting plots such as residual plots for each model could show the strengths and weaknesses of each and the different errors theyre prone to
5. Will the text in parts of the paper change at all e.g. when explaining algorithms etc or will we keep some the same/reworded
6. In Car Price paper, CV was not used to estimate train set performance, it was fit then predicted on the whole set. Confirm this

# New Features to test
* Link all customer reviews via extra datasets to perform NLP on that and join to this dataset
* Build or use transfer learning of a CV algorithm for listing photos 
* Get real estate data for suburbs via another source and compare to median/average or even CoreLogic price estimate
    * population density, average age and family type, new builds or infrastructure in locality
* knn pricing of closest 3 homes
* Host is based overseas
* Get council data for remodelling plans or past renos, land size, etc
* Additional house amenity features or building dimensions e.g. Average room size, porch size, pool size, Garage spaces etc
* Use CV algorithms to model how expensive the house looks from google map satellite and street images
* Distance-to... variables like subway, beach, major shopping centre
* Click through rates or other digital data collected from the website
    * expand out to number of photos, number of HD photos
    * wishlist additions
* Locality availibility  
* Other papers to analyse:
    * https://github.com/krishnaik06/Advanced-House-Price-Prediction-/blob/master/Feature%20Engineering.ipynb 
    * https://www.researchgate.net/publication/350616868_Airbnb_listings'_performance_determinants_and_predictive_models
    * https://scholarworks.calstate.edu/downloads/0p0968921
    * https://pdfs.semanticscholar.org/64d2/77ee8949d2eb5e5e14929d15ea008cb5b836.pdf
    * https://rankbreeze.com/airbnb-ranking-factors/



# Code Experiments
* Decision tree always gets 0 rmse on training dataset using default settings since it will overfit to infinite depth if learned, controlling max depth will reduce it e.g. try max depth = 3, 10, 30, 50 

```
dt = DecisionTreeRegressor(max_depth=32)
dt.fit(x_train.fillna(0), y_train)
rmse = mean_squared_error(dt.predict(x_train.fillna(0)), y_train)**1/2
print(f'dt RMSE = {rmse}')
``` 

* 

