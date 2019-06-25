require "csv"
require "ruby-fann"

# load cleaned data - note that y_data is a 2D-array.
x_data = []
y_data = []

CSV.foreach("./data/admission.csv", :headers=>false){|row|
  x_data.push([row[0].to_f, row[1].to_f])
  y_data.push([row[2].to_i])
}

# set up training and test data
test_size_percentage = 20.0 # 20.0%
test_set_size = x_data.size * (test_size_percentage/100.to_f)

test_x_data = x_data[0 .. (test_set_size-1)]
test_y_data = y_data[0 .. (test_set_size-1)]

training_x_data = x_data[test_set_size .. x_data.size]
training_y_data = y_data[test_set_size .. y_data.size]

# set up trainin data  model
train = RubyFann::TrainData.new(:inputs=>training_x_data, :desired_outputs=>training_y_data); #note that desired output must be a 2D array even if you have only one output node, in order to support networks with multiple output nodes.

# set up model / neural network and train using training data
model = RubyFann::Standard.new(
  num_inputs: 2,
  hidden_neurons: [6],
  num_outputs: 1)

# 5000 max_epochs, 500 errors between reports, and 0.01 desired
# mean-squared-error (i.e. stop if the error drops below 0.01)
model.train_on_data(train, 5000, 500, 0.01)

# predict single class
prediction = model.run([45, 85])
# round the output to get the prediction
puts "Algorithm predicted class: #{prediction.map{|e| e.round }}"

predicted=[]
test_x_data.each do |params|
  predicted.push(model.run(params).map{ |e| e.round} )
end

correct = predicted.collect.with_index{ |e,i| (e == test_y_data[i]) ? 1 : 0}.inject{|sum,e| sum+e}

puts "Accuracy: #{((correct.to_f / test_set_size) * 100).round(2)}% - test set of size #{test_size_percentage}%"

