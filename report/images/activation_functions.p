# set term x11 persist
set term png enhanced
set output "activation_functions.png"

# set size ratio 1
set xlabel 'x'
set ylabel '{/Symbol s}(x)'
set xrange [-6:6]
set yrange [-1.5:2]
set xzeroaxis
set yzeroaxis

relu(x) = x>0 ? x : 0
plot 1/(1+exp(-x)) title 'Logistic' lt rgb "#115EA6" linewidth 2, tanh(x) title 'tanh' lt rgb "red" linewidth 2, relu(x) title 'ReLU' lt rgb "dark-green" linewidth 2
