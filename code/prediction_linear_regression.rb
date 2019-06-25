# prediction_linear_regression.rb

require "csv"
require "ruby_linear_regression"

x_data = []
y_data = []

# push cleaned data into two arrays, one for the dependent variable, the
# other for the independent variable.
CSV.foreach("./data/staten-island-single-family-home-sales-2015.csv", :headers=> true){ |row|
  x_data.push( [row[0].to_i, row[1].to_i] )
  y_data.push( row[2].to_i )
}

# create an instance of the LinearRegression object and give it the data
linear_regression = RubyLinearRegression.new
linear_regression.load_training_data(x_data, y_data)

# train the linear regression model
linear_regression.train_normal_equation

# test
prediction_data = [2000, 1500]
predicted_price = linear_regression.predict(prediction_data)
puts "Predicted selling price for a 1500 sq foot house on a 2000 sq foot property: $#{predicted_price.round}"


