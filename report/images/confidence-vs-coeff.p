# set term x11 persist
set term png enhanced
set output "confidence-vs-coeff.png"

# set size ratio 1
set xlabel 'c'
set ylabel 'GoogLeNet Confidence'
# set xrange [-6:6]
set yrange [-0.05:1.2]
# set xzeroaxis
# set yzeroaxis

plot 'confidence-vs-coeff.dat' using 1:2 lt rgb "#115EA6" linewidth 2 title 'Acorn', 'confidence-vs-coeff.dat' using 1:3 lt rgb "dark-green" linewidth 2 title 'Dresser'
