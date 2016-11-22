set term png enhanced

set logscale x
set yrange [-0.1:1.2]

set xlabel 'Input dimensionality'
set ylabel 'Neuron output'

set output "ae_explanation_output.png"
plot 'ae_explanation_data.dat' using 1:5 lt rgb "#115EA6" title 'Original', 'ae_explanation_data.dat' using 1:6 lt rgb "dark-green" title 'Adversarial'



unset yrange
set ylabel 'Inner product'
set output "ae_explanation_ip.png"
plot 'ae_explanation_data.dat' using 1:3 lt rgb "#115EA6" title 'Original', 'ae_explanation_data.dat' using 1:4 lt rgb "dark-green" title 'Adversarial'
