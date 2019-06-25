# classification_linear_logistic_regression.rb

require "csv"
require "liblinear"

x_data = []
y_data = []

# load cleaned data
CSV.foreach("./data/admission.csv", :headers => false){|row|
  x_data.push([row[0].to_f, row[1].to_f]) # exam results in x
  y_data.push(row[2].to_i) # admission status in y
}

# split the data into a training data set and a new data set
# a good split is 80% data for training.

test_size_percentage = 20.0
test_set_size = x_data.size * (test_size_percentage/100.to_f)

test_x_data = x_data[0 .. (test_set_size-1)]
test_y_data = y_data[0 .. (test_set_size-1)]

training_x_data = x_data[test_set_size .. x_data.size]
training_y_data = y_data[test_set_size .. y_data.size]

# set up model and train
model = Liblinear.train(
  { solver_type: Liblinear::L2R_LR }, # L2-regularized logistic regression
  training_y_data, # training data classification
  training_x_data, # training data independent variables
  100              # bias
)

# predict class
prediction = Liblinear.predict(model, [45, 85])

# get prediction probabilities
probs = Liblinear.predict_probabilities(model, [45, 85])
probs = probs.sort

puts "Algorithm predicted class #{prediction}"
puts "#{probs[1]*100.round(2)}% probability of prediction"
puts "#{probs[0]*100.round(2)}% probability of being other class"

predicted = []
test_x_data.each do |params|
  predicted.push(Liblinear.predict(model, params))
end

correct = predicted.collect.with_index{|e,i| (e == test_y_data[i]) ? 1 : 0}.inject{ |sum,e| sum+e}

puts "Accuracy: #{((correct.to_f / test_set_size) * 100).round(2)}% - test set of size #{test_size_percentage}%"

