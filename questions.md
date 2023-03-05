# Questions during modelling 

1. during cross val, some folds have extremely high error (only for a couple models though)
2. what base features should we use for illustration of feature engineering importance?
3. should we only include tuned models in the final table or stick to default params or semi default
4. removal of outliers
5. what is a suitable number of plots to include? do papers have appendix for extra?
    * some very interesting plots such as residual plots for each model could show the strengths and weaknesses of each and the different errors theyre prone to


# Code Experiments
* Decision tree always gets 0 rmse on training dataset using default settings since it will overfit to infinite depth if learned, controlling max depth will reduce it e.g. try max depth = 3, 10, 30, 50 

```
dt = DecisionTreeRegressor(max_depth=32)
dt.fit(x_train.fillna(0), y_train)
rmse = mean_squared_error(dt.predict(x_train.fillna(0)), y_train)**1/2
print(f'dt RMSE = {rmse}')
``` 

* 

